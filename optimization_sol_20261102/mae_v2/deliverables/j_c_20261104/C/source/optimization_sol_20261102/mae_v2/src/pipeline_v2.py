"""Bounded, label-blind Nutrition5k parent/component inference.

A uses one RGB image, a parent layout, one selection and direct parent masses.
B reuses A's original layout, preserves parent IDs/bboxes/region ownership, and
reviews components once using the same RGB plus at most four 10%-expanded crops.
C reuses an identity and estimates total mass times parent and component ratios.
D preserves structure AND raw leaf grams: one code review of EVERY leaf in every
complete input image, optionally one new retrieval and one final selection.
Lexical conflict flags are audit evidence, never a per-image review gate.
E replaces supported leaf masses with multiples
of complete generic catalogue portions. Neither D nor E composes with the other.

Only leaves are nutrient-accounting entities. Composite parents have exactly one
leaf; decomposed parents have 2--6. There are at most 8 parents and 20 leaves.
Every prompt is built from explicitly supplied pixels, generated descriptions and
generic catalogue entries. No label, sample identifier or training example is
accepted. Provider retries, supervised calibration and whole-dish median fallback
belong to the runner. Each model stage permits one FORMAT-only repair.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re

from PIL import Image

from opt_pipeline import (ContractError, Pipeline, SYSTEM, TARGETS, NUTRIENTS,
                          _code, _json, _number, _object, _text, encode_image)

PIPELINE_VERSION = "components-v2.2"
SCHEMA_VERSION = 2
MAX_PARENTS, MAX_CHILDREN, MAX_LEAVES = 8, 6, 20
MAX_CROPS, CROP_PADDING, CANDIDATE_CAP = 4, .10, 32

LAYOUT_FORMAT = ('{"coverage_complete":true,"parents":[{"parent_id":"p1",'
    '"description":"visible dish","bbox":[0.1,0.1,0.9,0.9],"region_ids":["r1"],'
    '"accounting":"components","components":[{"component_id":"p1c1",'
    '"description":"visible cooked rice","broad_query":"cooked rice"},'
    '{"component_id":"p1c2","description":"visible chicken",'
    '"broad_query":"cooked chicken"}]}]}')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _image_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bbox(value):
    if not isinstance(value, list) or len(value) != 4:
        raise ContractError("invalid_bbox")
    box = [_number(n) for n in value]
    if any(n > 1 for n in box) or box[0] >= box[2] or box[1] >= box[3]:
        raise ContractError("invalid_bbox")
    return box


def _positive(value):
    number = _number(value)
    if number <= 0:
        raise ContractError("nonpositive_mass_or_ratio")
    return number


def _normalize(values):
    values = [_number(n) for n in values]
    maximum = max(values, default=0)
    if maximum == 0:
        raise ContractError("zero_ratio_sum")
    scaled = [n / maximum for n in values]
    denominator = sum(scaled)
    return [n / denominator for n in scaled]


def _families(text):
    words = set(re.findall(r"[a-z]+", text.lower()))
    groups = {"chicken": {"chicken", "poultry"}, "beef": {"beef", "steak"},
              "pork": {"pork", "ham", "bacon"}, "fish": {"fish", "salmon", "tuna", "cod"},
              "tofu": {"tofu"}, "egg": {"egg", "eggs"}, "rice": {"rice"},
              "pasta": {"pasta", "noodles", "spaghetti", "macaroni"}}
    return sorted(name for name, tokens in groups.items() if words & tokens)


def relation_hints(description, category=""):
    """Conservative lexical hypotheses, NOT verified food ontology facts."""
    text = (description + " " + category).lower()
    mixed = bool(re.search(r"\b(salad|sandwich|pizza|casserole|burrito|taco|soup|stew|"
                           r"lasagna|risotto|mixed dishes?|rice with|with rice)\b", text))
    dressing = ("without" if re.search(r"\b(no|without) dressing\b", text) else
                "with" if re.search(r"\b(with dressing|dressing visible|visibly dressed)\b", text) else "unknown")
    return {"inferred_not_verified": True, "family_hints": _families(description),
            "accounting_hint": "likely_composite" if mixed else "unspecified",
            "preparation_hints": [v for v in ("fried", "grilled", "roasted", "boiled", "steamed")
                                  if re.search(r"\b" + v + r"\b", text)], "dressing_hint": dressing}


def _parse_layout(raw):
    data = _object(_json(raw), {"coverage_complete", "parents"})
    if not isinstance(data["coverage_complete"], bool) or not isinstance(data["parents"], list):
        raise ContractError("invalid_layout")
    if not 0 <= len(data["parents"]) <= MAX_PARENTS or (data["coverage_complete"] and not data["parents"]):
        raise ContractError("invalid_parent_count")
    parents, ids, regions, leaves = [], set(), set(), set()
    for parent in data["parents"]:
        _object(parent, {"parent_id", "description", "bbox", "region_ids", "accounting", "components"})
        pid = parent["parent_id"]
        if not isinstance(pid, str) or re.fullmatch(r"p[1-8]", pid) is None or pid in ids:
            raise ContractError("invalid_parent_id")
        owned = parent["region_ids"]
        if (not isinstance(owned, list) or not owned or any(not isinstance(x, str) or
                re.fullmatch(r"r[1-9]\d*", x) is None for x in owned)
                or len(set(owned)) != len(owned) or set(owned) & regions):
            raise ContractError("overlapping_or_invalid_parent_regions")
        mode, children = parent["accounting"], parent["components"]
        if not isinstance(mode, str) or mode not in {"composite", "components"} or not isinstance(children, list):
            raise ContractError("invalid_accounting_mode")
        if (mode == "composite" and len(children) != 1) or (mode == "components" and not 2 <= len(children) <= MAX_CHILDREN):
            raise ContractError("invalid_component_count")
        parsed_children, descriptions = [], set()
        for child in children:
            _object(child, {"component_id", "description", "broad_query"})
            cid = child["component_id"]
            if not isinstance(cid, str) or re.fullmatch(pid + r"c[1-6]", cid) is None or cid in leaves:
                raise ContractError("invalid_component_id")
            description, query = _text(child["description"]), _text(child["broad_query"])
            normalized = " ".join(re.findall(r"[a-z0-9]+", description.lower()))
            if normalized in descriptions:
                raise ContractError("duplicate_component_description")
            descriptions.add(normalized)
            leaves.add(cid)
            parsed_children.append({"component_id": cid, "description": description, "broad_query": query})
        if {c["component_id"] for c in parsed_children} != {pid + "c" + str(i + 1) for i in range(len(children))}:
            raise ContractError("nonconsecutive_component_ids")
        ids.add(pid)
        regions.update(owned)
        parents.append({"parent_id": pid, "description": _text(parent["description"]),
                        "bbox": _bbox(parent["bbox"]), "region_ids": list(owned), "accounting": mode,
                        "components": sorted(parsed_children, key=lambda c: c["component_id"])})
    if ids != {"p" + str(i + 1) for i in range(len(parents))}:
        raise ContractError("nonconsecutive_parent_ids")
    if len(leaves) > MAX_LEAVES:
        raise ContractError("too_many_leaves")
    return {"coverage_complete": data["coverage_complete"],
            "parents": sorted(parents, key=lambda p: p["parent_id"])}


def _flatten(parents):
    return [{**deepcopy(child), "item_id": child["component_id"], "parent_id": p["parent_id"],
             "parent_description": p["description"], "accounting": p["accounting"]}
            for p in parents for child in p["components"]]


class ComponentPipeline(Pipeline):
    """Public interfaces intentionally accept no sample metadata or labels."""

    @staticmethod
    def configuration(config=None):
        config = config or {}
        if config.get("candidate_cap", 32) != 32 or config.get("image_mode", "native") != "native":
            raise ContractError("v2_requires_native_cap32")
        crop = config.get("crop_review", False)
        if not isinstance(crop, bool):
            raise ContractError("invalid_crop_review")
        mode = config.get("portion_mode", "direct")
        if not isinstance(mode, str) or mode not in {"direct", "total_ratio"}:
            raise ContractError("invalid_component_portion_mode")
        return {"pipeline_version": PIPELINE_VERSION, "schema_version": SCHEMA_VERSION,
                "image_mode": "native", "candidate_cap": 32, "crop_review": crop,
                "portion_mode": mode, "max_parents": MAX_PARENTS, "max_children": MAX_CHILDREN,
                "max_leaves": MAX_LEAVES, "max_crops": MAX_CROPS, "crop_padding": CROP_PADDING,
                "max_schema_repairs": 1, "max_semantic_retrieval_rounds": 1,
                "semantic_review_scope": "all_leaves_all_complete_images", "positive_leaf_mass": True}

    def dependencies(self, config=None):
        config = self.configuration(config)
        return {"config": config, "config_hash": _digest(config), "catalog_hash": _digest(self.catalog),
                "implementation_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def _stage(self, name, prompt, images, parser, stages):
        if images is None:
            images = []
        if isinstance(images, str):
            images = [images]
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content":
            [{"type": "text", "text": prompt}] +
            [{"type": "image_url", "image_url": {"url": picture}} for picture in images]}]
        stage = {"stage": name, "attempts": [], "image_count": len(images)}
        stages.append(stage)
        for attempt in range(2):
            raw = self.chat(messages)
            entry = {"raw": raw, "valid": False}
            stage["attempts"].append(entry)
            try:
                parsed = parser(raw)
            except ContractError as exc:
                entry["error"] = str(exc)
                if attempt == 0:
                    messages += [{"role": "assistant", "content": raw}, {"role": "user", "content":
                        "Correct FORMAT only to satisfy the original contract. Preserve visual interpretation. "
                        "Do not reconsider accuracy. Return only the requested JSON. Contract error: " + str(exc)}]
            else:
                entry["valid"] = True
                stage["parsed"] = parsed
                return parsed
        return None

    def layout(self, image_path):
        stages = []
        prompt = ("Describe ALL visible edible parent dishes or separate sides. Their physical region ownership "
            "must be disjoint. Use stable consecutive parent IDs p1..p8 and region IDs r1... Do not count a parent "
            "and its components twice. A visually supported decomposition has accounting=components and 2--6 "
            "children; an unresolved or atomic food has accounting=composite and exactly one child. Across the "
            "plate use at most 20 children. Child IDs are parent ID plus c1,c2,... Give each child a specific "
            "description and a broader retrieval query. Do not add invisible ingredients, oil or sauces. A "
            "component may be spatially intermingled; its crop is not an extra food. Parent bbox is normalized "
            "[left,top,right,bottom] in THIS image. If you cannot cover all visible foods within the contract, "
            "set coverage_complete=false; never omit food and claim completeness. Return ONLY this JSON "
            "schema, no extra fields: " + LAYOUT_FORMAT)
        parsed = self._stage("layout", prompt, encode_image(image_path, "native"), _parse_layout, stages)
        complete = parsed is not None and parsed["coverage_complete"]
        return self._result(stages, status="ok" if complete else "failed",
            failure=None if complete else "layout_contract" if parsed is None else "incomplete_visible_coverage",
            parents=parsed["parents"] if parsed else [], coverage_complete=bool(complete),
            input_image_sha256=_image_hash(image_path), pipeline_version=PIPELINE_VERSION)

    def _checked_layout(self, image_path, saved):
        if not isinstance(saved, dict) or saved.get("status") != "ok":
            raise ContractError("missing_complete_layout")
        if saved.get("input_image_sha256") != _image_hash(image_path):
            raise ContractError("layout_image_mismatch")
        parsed = _parse_layout(json.dumps({"coverage_complete": saved.get("coverage_complete"),
                                         "parents": saved.get("parents")}))
        if not parsed["coverage_complete"]:
            raise ContractError("incomplete_visible_coverage")
        return parsed["parents"]

    @staticmethod
    def _crop_priority(parent):
        text = " ".join([parent["description"]] + [c["description"] for c in parent["components"]]).lower()
        ambiguity = bool(re.search(r"\b(uncertain|possibly|possible|or|unclear)\b", text))
        protein = bool(set(_families(text)) & {"chicken", "beef", "pork", "fish", "tofu", "egg"})
        sauce = "sauce" in text or "dressing" in text
        b = parent["bbox"]
        return (-int(ambiguity), -int(parent["accounting"] == "components"), -int(protein),
                -int(sauce), -(b[2] - b[0]) * (b[3] - b[1]), parent["parent_id"])

    def _images_and_crops(self, image_path, parents):
        images, crops = [encode_image(image_path, "native")], []
        with Image.open(image_path) as source:
            source = source.convert("RGB")
            width, height = source.size
            for parent in sorted(parents, key=self._crop_priority)[:MAX_CROPS]:
                x0, y0, x1, y1 = parent["bbox"]
                dx, dy = (x1 - x0) * CROP_PADDING, (y1 - y0) * CROP_PADDING
                bounds = (max(0, math.floor((x0 - dx) * width)), max(0, math.floor((y0 - dy) * height)),
                          min(width, math.ceil((x1 + dx) * width)), min(height, math.ceil((y1 + dy) * height)))
                crop = source.crop(bounds)
                crop.thumbnail((1024, 1024), Image.Resampling.BICUBIC)
                clean = Image.new("RGB", crop.size)
                clean.paste(crop)
                out = BytesIO()
                clean.save(out, format="PNG")
                payload = out.getvalue()
                images.append("data:image/png;base64," + base64.b64encode(payload).decode("ascii"))
                crops.append({"crop_id": "crop_" + parent["parent_id"], "parent_id": parent["parent_id"],
                              "pixel_bounds": list(bounds), "pixels_sha256": hashlib.sha256(payload).hexdigest(),
                              "size": list(clean.size), "padding_fraction": CROP_PADDING})
        return images, crops

    def _retrieve(self, leaves, stages, stage_name="retrieval"):
        queries = [query for leaf in leaves for query in (leaf["description"], leaf["broad_query"])]
        vectors = []
        for start in range(0, len(queries), 12):
            batch = self.embed(queries[start:start + 12])
            if len(batch) != len(queries[start:start + 12]):
                raise ContractError("embedding_count_mismatch")
            vectors.extend(batch)
        candidates = {}
        for i, leaf in enumerate(leaves):
            codes = self.retriever.hybrid(queries[2 * i:2 * i + 2], vectors[2 * i:2 * i + 2], cap=32)
            if not codes or len(codes) > 32 or len(set(codes)) != len(codes) or any(c not in self.foods for c in codes):
                raise ContractError("invalid_candidate_set")
            candidates[leaf["component_id"]] = codes
        stages.append({"stage": stage_name, "attempts": [], "queries": queries,
                       "codes_by_item": deepcopy(candidates), "candidate_cap": 32})
        return candidates

    def _candidates(self, codes):
        result = []
        for code in codes:
            food = self.foods[code]
            result.append({"code": code, "description": _text(food["description"]),
                "density_per_100g": {key: _number(food["density"][key]) for key in NUTRIENTS},
                "relationship_hints": relation_hints(food["description"], food.get("category", ""))})
        return result

    @staticmethod
    def _parse_selection(raw, leaves, candidates):
        data = _object(_json(raw), {"items"})
        expected = {leaf["component_id"] for leaf in leaves}
        if not isinstance(data["items"], list) or len(data["items"]) != len(expected):
            raise ContractError("missing_or_extra_selection")
        selected = {}
        for row in data["items"]:
            _object(row, {"component_id", "primary_code"})
            cid = row["component_id"]
            if not isinstance(cid, str) or cid not in expected or cid in selected:
                raise ContractError("unknown_or_duplicate_selection")
            code = _code(row["primary_code"])
            if code not in candidates[cid]:
                raise ContractError("selected_code_outside_candidates")
            selected[cid] = code
        return [{**deepcopy(leaf), "primary_code": selected[leaf["component_id"]], "alternative_codes": []}
                for leaf in leaves]

    def _select(self, leaves, candidates, images, stages, name="select"):
        context = [{"component_id": leaf["component_id"], "description": leaf["description"],
                    "parent_description": leaf["parent_description"], "accounting": leaf["accounting"],
                    "candidates": self._candidates(candidates[leaf["component_id"]])} for leaf in leaves]
        prompt = ("Select exactly one candidate code for every fixed accounting leaf. Preserve all leaf IDs "
            "and the physical interpretation; do not add, merge, split or change quantities. A standalone "
            "ingredient must not become an entire mixed dish. Catalogue relation hints are lexical inferences, "
            "not verified facts. Use visual food/preparation evidence first. Nutrient density is generic "
            "catalogue data, never a target to optimize. Do not choose higher protein merely because it is "
            "higher. Unknown dressing is not evidence of no dressing. Return ONLY JSON "
            '{"items":[{"component_id":"p1c1","primary_code":"12345678"}]}. Context: ' + json.dumps(context))
        return self._stage(name, prompt, images,
                           lambda raw: self._parse_selection(raw, leaves, candidates), stages)

    def identify(self, image_path, config, layout=None):
        cfg = self.configuration(config)
        initial = deepcopy(layout) if layout is not None else self.layout(image_path)
        stages = deepcopy(initial.get("stages", []))
        if initial.get("status") != "ok":
            return self._result(stages, status="failed", failure=initial.get("failure", "layout_failed"),
                parents=[], items=[], leaves=[], final_codes=[], selected_codes=[], retrieved_codes=[],
                candidates={}, layout_result=initial, config=cfg, pipeline_version=PIPELINE_VERSION)
        parents = self._checked_layout(image_path, initial)
        images, crops = [encode_image(image_path, "native")], []
        if cfg["crop_review"]:
            images, crops = self._images_and_crops(image_path, parents)
            fixed = {p["parent_id"]: (p["bbox"], p["region_ids"]) for p in parents}
            def parse_review(raw):
                data = _parse_layout(raw)
                actual = {p["parent_id"]: (p["bbox"], p["region_ids"]) for p in data["parents"]}
                if actual != fixed:
                    raise ContractError("crop_review_changed_parent_ownership")
                return data
            prompt = ("Review the components using the original image followed by crops from that SAME image. "
                "These are not additional foods or views. Preserve every original parent ID, bbox and region "
                "ownership exactly. You may correct descriptions, component identities and accounting mode. "
                "Use at most 6 children per parent and 20 total. No invisible ingredients. If coverage cannot "
                "be maintained, set coverage_complete=false. Return ONLY the complete layout schema: " +
                LAYOUT_FORMAT + " Original layout: " + json.dumps({"coverage_complete": True, "parents": parents}) +
                " Crop order after the original: " + json.dumps(crops))
            revised = self._stage("crop_review", prompt, images, parse_review, stages)
            if revised is None or not revised["coverage_complete"]:
                return self._result(stages, status="failed", failure="crop_review_contract" if revised is None
                    else "incomplete_visible_coverage", parents=[], items=[], leaves=[], candidates={},
                    final_codes=[], selected_codes=[], retrieved_codes=[], layout_result=initial, crops=crops,
                    config=cfg, pipeline_version=PIPELINE_VERSION)
            parents = revised["parents"]
        leaves = _flatten(parents)
        candidates = self._retrieve(leaves, stages)
        selected = self._select(leaves, candidates, images, stages)
        retrieved = list(dict.fromkeys(c for group in candidates.values() for c in group))
        if selected is None:
            return self._result(stages, status="failed", failure="selection_contract", parents=parents,
                leaves=[], items=[], candidates=candidates, retrieved_codes=retrieved, final_codes=[],
                selected_codes=[], layout_result=initial, crops=crops, config=cfg, pipeline_version=PIPELINE_VERSION)
        final = list(dict.fromkeys(leaf["primary_code"] for leaf in selected))
        return self._result(stages, status="ok", failure=None, parents=parents, leaves=selected,
            items=deepcopy(selected), candidates=candidates, retrieved_codes=retrieved, final_codes=final,
            selected_codes=final, description="; ".join(p["description"] for p in parents),
            layout_result=initial, crops=crops, input_image_sha256=_image_hash(image_path), config=cfg,
            pipeline_version=PIPELINE_VERSION)

    def _checked_identity(self, image_path, identity):
        if not isinstance(identity, dict) or identity.get("status") != "ok":
            raise ContractError("missing_identity")
        if identity.get("input_image_sha256") != _image_hash(image_path):
            raise ContractError("identity_image_mismatch")
        parents = _parse_layout(json.dumps({"coverage_complete": True, "parents": identity.get("parents")}))["parents"]
        flat = _flatten(parents)
        supplied = identity.get("leaves")
        if not isinstance(supplied, list) or len(supplied) != len(flat):
            raise ContractError("missing_identity_leaf")
        indexed = {}
        for leaf in supplied:
            cid = leaf.get("component_id") if isinstance(leaf, dict) else None
            if not isinstance(cid, str) or cid in indexed:
                raise ContractError("invalid_identity_leaf")
            indexed[cid] = leaf
        for leaf in flat:
            saved = indexed.get(leaf["component_id"], {})
            if any(saved.get(key) != value for key, value in leaf.items()):
                raise ContractError("identity_leaf_structure_mismatch")
            code = _code(saved.get("primary_code"))
            if code not in self.foods or code not in identity.get("candidates", {}).get(leaf["component_id"], []):
                raise ContractError("identity_code_outside_candidates")
            leaf.update(primary_code=code, alternative_codes=[])
        return parents, flat

    @staticmethod
    def _parse_masses(raw, parents, mode):
        data = _object(_json(raw), {"parents", "total_grams"} if mode == "total_ratio" else {"parents"})
        rows = data["parents"]
        if not isinstance(rows, list) or len(rows) != len(parents):
            raise ContractError("missing_or_extra_mass_parent")
        expected, by_id = {p["parent_id"]: p for p in parents}, {}
        field = "ratio" if mode == "total_ratio" else "grams"
        for row in rows:
            _object(row, {"parent_id", field, "components"})
            pid = row["parent_id"]
            if not isinstance(pid, str) or pid not in expected or pid in by_id:
                raise ContractError("invalid_mass_parent_id")
            children = row["components"]
            expected_children = {c["component_id"] for c in expected[pid]["components"]}
            if not isinstance(children, list) or len(children) != len(expected_children):
                raise ContractError("missing_mass_component")
            ratios = {}
            for child in children:
                _object(child, {"component_id", "ratio"})
                cid = child["component_id"]
                if not isinstance(cid, str) or cid not in expected_children or cid in ratios:
                    raise ContractError("invalid_mass_component_id")
                ratios[cid] = _positive(child["ratio"])
            _normalize(list(ratios.values()))
            by_id[pid] = {field: _positive(row[field]), "ratios": ratios}
        if mode == "total_ratio":
            total = _positive(data["total_grams"])
            fractions = _normalize([by_id[p["parent_id"]]["ratio"] for p in parents])
            parent_masses = {p["parent_id"]: total * fraction for p, fraction in zip(parents, fractions)}
        else:
            parent_masses = {p["parent_id"]: by_id[p["parent_id"]]["grams"] for p in parents}
        masses = {}
        for parent in parents:
            ratios = by_id[parent["parent_id"]]["ratios"]
            fractions = _normalize([ratios[c["component_id"]] for c in parent["components"]])
            for child, fraction in zip(parent["components"], fractions):
                masses[child["component_id"]] = _positive(parent_masses[parent["parent_id"]] * fraction)
        return {"leaf_grams": masses, "parent_grams": parent_masses}

    def _aggregate(self, identity, masses, stages, **extra):
        totals = dict.fromkeys(TARGETS, 0.0)
        portions, parent_mass = [], {p["parent_id"]: 0.0 for p in identity["parents"]}
        expected = {leaf["component_id"] for leaf in identity["leaves"]}
        if set(masses) != expected:
            raise ContractError("incomplete_leaf_masses")
        for leaf in identity["leaves"]:
            cid, code = leaf["component_id"], leaf["primary_code"]
            grams = _positive(masses[cid])
            density = {key: _number(self.foods[code]["density"][key]) for key in NUTRIENTS}
            contribution = self.prediction_for_weight(code, grams)
            for key in TARGETS:
                totals[key] = _number(totals[key] + contribution[key])
            parent_mass[leaf["parent_id"]] = _number(parent_mass[leaf["parent_id"]] + grams)
            portions.append({"item_id": cid, "component_id": cid, "parent_id": leaf["parent_id"],
                             "grams": grams, "code": code, "density": density})
        return self._result(stages, status="ok", failure=None, prediction=totals,
            raw_prediction=deepcopy(totals), portions=portions, leaves=deepcopy(portions),
            raw_leaves=deepcopy(portions), parents=[{"parent_id": pid, "grams": value}
                                                  for pid, value in parent_mass.items()],
            pipeline_version=PIPELINE_VERSION, **extra)

    def portion(self, image_path, identity, config):
        cfg, stages = self.configuration(config), []
        if identity.get("status") != "ok":
            return self._result(stages, status="failed", failure="identity_failed", prediction=None,
                                raw_prediction=None, portions=[], leaves=[], raw_leaves=[])
        parents, leaves = self._checked_identity(image_path, identity)
        mode = cfg["portion_mode"]
        context = [{"parent_id": p["parent_id"], "description": p["description"], "accounting": p["accounting"],
                    "components": [{"component_id": c["component_id"], "description": c["description"],
                        "selected_food": self.foods[next(l["primary_code"] for l in leaves
                                                        if l["component_id"] == c["component_id"])]["description"]}
                                   for c in p["components"]]} for p in parents]
        contract = ('{"total_grams":300,"parents":[{"parent_id":"p1","ratio":1,' if mode == "total_ratio" else
                    '{"parents":[{"parent_id":"p1","grams":300,')
        contract += '"components":[{"component_id":"p1c1","ratio":1}]}]}'
        prompt = ("Estimate cooked edible mass from the original RGB image. Fixed parent and component identities "
            "must not change. Cover every parent and child exactly once; do not invent hidden ingredients or "
            "infer a measured plate diameter. " + ("Estimate total plate grams and relative parent masses. "
            if mode == "total_ratio" else "Estimate the grams of each parent. ") +
            "For each parent estimate relative component masses, not pixel-area fractions. The program "
            "normalizes ratios within the parent; composite parents have one child with ratio 1. All values "
            "must be finite and strictly positive. Every visible leaf must receive positive grams; zero "
            "must not remove an uncertain component. Return ONLY JSON: " + contract +
            " Fixed identities: " + json.dumps(context))
        parsed = self._stage("portion_" + mode, prompt, encode_image(image_path, "native"),
                             lambda raw: self._parse_masses(raw, parents, mode), stages)
        if parsed is None:
            return self._result(stages, status="failed", failure="portion_contract", prediction=None,
                                raw_prediction=None, portions=[], leaves=[], raw_leaves=[])
        return self._aggregate(identity, parsed["leaf_grams"], stages, method=mode)

    def semantic_conflicts(self, identity):
        """Label-free diagnostic flags; they never gate individual D reviews."""
        conflicts = []
        for leaf in identity.get("leaves", []):
            code = leaf.get("primary_code")
            if code not in self.foods:
                continue
            visual = relation_hints(leaf["description"])
            food = self.foods[code]
            candidate = relation_hints(food["description"], food.get("category", ""))
            reasons = []
            if ((leaf["accounting"] == "components" or visual["family_hints"])
                    and visual["accounting_hint"] != "likely_composite"
                    and candidate["accounting_hint"] == "likely_composite"):
                reasons.append("ingredient_selected_as_composite")
            if (visual["family_hints"] and candidate["family_hints"] and
                    not set(visual["family_hints"]) & set(candidate["family_hints"])):
                reasons.append("different_explicit_food_family")
            if {visual["dressing_hint"], candidate["dressing_hint"]} == {"with", "without"}:
                reasons.append("explicit_dressing_conflict")
            if reasons:
                conflicts.append({"component_id": leaf["component_id"], "code": code, "reasons": reasons,
                                  "inferred_not_verified": True})
        return conflicts

    def _checked_base_masses(self, identity, result):
        if result.get("status") != "ok" or result.get("prediction") is None:
            raise ContractError("missing_complete_base_portion")
        rows = result.get("raw_leaves", result.get("portions"))
        if not isinstance(rows, list) or len(rows) != len(identity["leaves"]):
            raise ContractError("invalid_base_portions")
        by_id = {leaf["component_id"]: leaf for leaf in identity["leaves"]}
        masses = {}
        for row in rows:
            cid = row.get("component_id", row.get("item_id"))
            if not isinstance(cid, str) or cid not in by_id or cid in masses or row.get("code") != by_id[cid]["primary_code"]:
                raise ContractError("base_portion_identity_mismatch")
            masses[cid] = _positive(row.get("grams"))
        return masses

    @staticmethod
    def _method_fallback(base, method, reason, stages, **extra):
        result = deepcopy(base)
        result["stages"] = deepcopy(base.get("stages", [])) + deepcopy(stages)
        result["repaired"] = bool(base.get("repaired") or any(len(s["attempts"]) > 1 and
                    s["attempts"][-1]["valid"] for s in stages))
        result.update(method=method, method_fallback=True, method_fallback_reason=reason, **extra)
        return result

    def review_codes(self, image_path, identity, portion_result, config):
        self.configuration(config)
        self._checked_identity(image_path, identity)
        masses = self._checked_base_masses(identity, portion_result)
        if portion_result.get("method") in {"standard_portions", "semantic_review"}:
            raise ContractError("second_round_composition_forbidden")
        conflicts = self.semantic_conflicts(identity)
        stages, affected = [], {leaf["component_id"] for leaf in identity["leaves"]}
        leaves = list(identity["leaves"])
        candidates = {leaf["component_id"]: identity["candidates"][leaf["component_id"]] for leaf in leaves}
        context = [{"component_id": l["component_id"], "description": l["description"],
                    "current_code": l["primary_code"], "candidates": self._candidates(candidates[l["component_id"]])}
                   for l in leaves]
        def parse_review(raw):
            data = _object(_json(raw), {"items"})
            if not isinstance(data["items"], list) or len(data["items"]) != len(leaves):
                raise ContractError("missing_review_item")
            seen, result = set(), []
            for row in data["items"]:
                _object(row, {"component_id", "action", "primary_code", "query", "broad_query"})
                cid, action = row["component_id"], row["action"]
                if not isinstance(cid, str) or cid not in affected or cid in seen:
                    raise ContractError("invalid_review_item")
                if not isinstance(action, str) or action not in {"select", "retrieve"}:
                    raise ContractError("invalid_review_action")
                if action == "select":
                    code = _code(row["primary_code"])
                    if code not in candidates[cid] or row["query"] is not None or row["broad_query"] is not None:
                        raise ContractError("invalid_review_selection")
                    result.append({"component_id": cid, "action": action, "primary_code": code})
                else:
                    if row["primary_code"] is not None:
                        raise ContractError("retrieve_cannot_select_code")
                    result.append({"component_id": cid, "action": action,
                                   "query": _text(row["query"]), "broad_query": _text(row["broad_query"])})
                seen.add(cid)
            return result
        prompt = ("Review EVERY selected food code against the original RGB, including components with no "
            "diagnostic flag. Check food identity, whole-dish versus ingredient meaning, preparation and "
            "dressing against visual evidence. Flags and catalogue relations are unverified lexical hints; "
            "visual evidence takes priority. Preserve physical leaf "
            "identity, parent structure and grams. Choose an existing compatible code, retaining the current "
            "one when uncertain. Only if the candidates have no compatible code, action=retrieve may provide "
            "two concise queries for the SAME visible identity. Do not invent unseen ingredients or optimize "
            "nutrient targets. Return ONLY JSON {\"items\":[{\"component_id\":\"p1c1\",\"action\":\"select\","
            "\"primary_code\":\"12345678\",\"query\":null,\"broad_query\":null}]}. For retrieve set "
            "primary_code=null and both queries to strings. Flags: " + json.dumps(conflicts) + " Context: " + json.dumps(context))
        reviewed = self._stage("semantic_review", prompt, encode_image(image_path, "native"), parse_review, stages)
        if reviewed is None:
            return deepcopy(identity), self._method_fallback(portion_result, "semantic_review", "review_contract", stages,
                                                             semantic_conflicts=conflicts, changed_codes=[])
        new_identity = deepcopy(identity)
        codes = {row["component_id"]: row["primary_code"] for row in reviewed if row["action"] == "select"}
        needs = [row for row in reviewed if row["action"] == "retrieve"]
        if needs:
            by_id = {leaf["component_id"]: leaf for leaf in leaves}
            search_leaves = [{**by_id[row["component_id"]], "description": row["query"], "broad_query": row["broad_query"]}
                             for row in needs]
            fresh = self._retrieve(search_leaves, stages, "semantic_retrieval_once")
            actual_leaves = [by_id[row["component_id"]] for row in needs]
            selected = self._select(actual_leaves, fresh, encode_image(image_path, "native"), stages, "semantic_final_select")
            if selected is None:
                return deepcopy(identity), self._method_fallback(portion_result, "semantic_review", "final_selection_contract", stages,
                                                                 semantic_conflicts=conflicts, changed_codes=[])
            codes.update({leaf["component_id"]: leaf["primary_code"] for leaf in selected})
            new_identity["candidates"].update(fresh)  # Replace, never append a second 32-code set.
        changed = []
        for leaf in new_identity["leaves"]:
            cid = leaf["component_id"]
            if cid in codes and codes[cid] != leaf["primary_code"]:
                changed.append({"component_id": cid, "old_code": leaf["primary_code"], "new_code": codes[cid]})
                leaf["primary_code"] = codes[cid]
        new_identity["items"] = deepcopy(new_identity["leaves"])
        new_identity["final_codes"] = list(dict.fromkeys(l["primary_code"] for l in new_identity["leaves"]))
        new_identity["selected_codes"] = list(new_identity["final_codes"])
        new_identity["retrieved_codes"] = list(dict.fromkeys(c for group in new_identity["candidates"].values() for c in group))
        new_identity["stages"] = deepcopy(identity.get("stages", [])) + deepcopy(stages)
        new_identity["repaired"] = bool(identity.get("repaired") or any(len(s["attempts"]) > 1 and
            s["attempts"][-1]["valid"] for s in stages))
        new_identity["semantic_review_applied"] = True
        result = self._aggregate(new_identity, masses, deepcopy(portion_result.get("stages", [])) + stages,
            method="semantic_review", method_fallback=False, method_fallback_reason=None,
            semantic_conflicts=conflicts, changed_codes=changed)
        return new_identity, result

    def usable_standard_options(self, code):
        """Only explicit physical units; no label-derived units or invented grams."""
        result, seen = [], set()
        for option in self.catalog.get("standard_options", {}).get(code, []):
            if not isinstance(option, dict):
                raise ContractError("invalid_standard_option")
            identifier, description, grams = _text(option.get("option_id")), _text(option.get("description")), _number(option.get("grams"))
            text = description.lower()
            if identifier in seen:
                raise ContractError("duplicate_standard_option_id")
            seen.add(identifier)
            if (grams <= 0 or re.search(r"quantity not specified|guideline|\bunspecified\b|\bservings?\b|"
                                      r"\bnot specified\b", text)):
                continue
            # A count multiplies the COMPLETE description, including fractions.
            number = r"(?:\d+(?:\.\d+)?(?:[ -]\d+/\d+)?|\d+/\d+|\.\d+)"
            unit = r"(?:cups?|tablespoons?|teaspoons?|tbsp|tsp|pieces?|slices?|patt(?:y|ies)|"
            unit += r"eggs?|muffins?|rolls?|buns?|sandwich(?:es)?|tortillas?|biscuits?|cookies?|"
            unit += r"fillets?|drumsticks?|wings?|breasts?|thighs?|links?|meatballs?|pancakes?|waffles?|scoops?)"
            if re.match(r"^\s*" + number + r"\s+" + unit + r"\b", text) is None:
                continue
            result.append({"option_id": identifier, "description": description, "grams": grams})
        return result

    def standard_portion_coverage(self, identity):
        leaves = identity.get("leaves", [])
        eligible = [l["component_id"] for l in leaves if self.usable_standard_options(l["primary_code"])]
        return {"eligible_leaf_ids": eligible, "eligible_leaf_count": len(eligible), "total_leaf_count": len(leaves)}

    def standard_portions(self, image_path, identity, portion_result, config):
        self.configuration(config)
        self._checked_identity(image_path, identity)
        masses = self._checked_base_masses(identity, portion_result)
        if portion_result.get("method") in {"standard_portions", "semantic_review"} or identity.get("semantic_review_applied"):
            raise ContractError("second_round_composition_forbidden")
        eligible = {l["component_id"]: self.usable_standard_options(l["primary_code"]) for l in identity["leaves"]}
        eligible = {cid: opts for cid, opts in eligible.items() if opts}
        retained = sorted(set(masses) - set(eligible))
        source_hash = self.catalog.get("standard_portion_source_sha256")
        if not eligible:
            return self._method_fallback(portion_result, "standard_portions", "no_usable_standard_portion", [],
                catalog_eligible_leaf_count=0, catalog_retained_leaf_ids=retained, option_choices=[],
                standard_portion_source_sha256=source_hash)
        leaf_codes = {l["component_id"]: l["primary_code"] for l in identity["leaves"]}
        context = [{"component_id": l["component_id"], "description": l["description"],
                    "selected_food": self.foods[l["primary_code"]]["description"], "options": eligible[l["component_id"]]}
                   for l in identity["leaves"] if l["component_id"] in eligible]
        def parse_units(raw):
            data = _object(_json(raw), {"items"})
            if not isinstance(data["items"], list) or len(data["items"]) != len(eligible):
                raise ContractError("missing_or_extra_standard_portion")
            seen, result = set(), []
            for row in data["items"]:
                _object(row, {"component_id", "option_id", "count"})
                cid = row["component_id"]
                if not isinstance(cid, str) or cid not in eligible or cid in seen:
                    raise ContractError("invalid_standard_portion_id")
                identifier, count = _text(row["option_id"]), _positive(row["count"])
                option = next((o for o in eligible[cid] if o["option_id"] == identifier), None)
                if option is None:
                    raise ContractError("illegal_standard_option")
                result.append({"component_id": cid, "option_id": identifier, "count": count,
                               "option_description": option["description"], "option_grams": option["grams"],
                               "grams": _positive(count * option["grams"]), "code": leaf_codes[cid],
                               "source_refs": deepcopy(self.catalog.get("standard_option_sources", {})
                                   .get(leaf_codes[cid], {}).get(identifier, [])),
                               "source_sha256": source_hash})
                seen.add(cid)
            return result
        prompt = ("Estimate portions of the listed fixed visible components from the original RGB. Select one "
            "generic catalogue option and estimate its finite strictly positive count. Zero cannot remove "
            "a visible component. Count is the multiplier of the "
            "COMPLETE option description: if the option is '1/2 cup' and you see 1.5 cups, count=3. The program "
            "will multiply count by that option's grams. These are general food units, not labels for this "
            "image. Do not invent scale measurements. Do not change food identities or estimate additional "
            "components. Return every listed component exactly once, ONLY JSON "
            '{"items":[{"component_id":"p1c1","option_id":"s1","count":2.5}]}. Context: ' + json.dumps(context))
        stages = []
        choices = self._stage("standard_portions", prompt, encode_image(image_path, "native"), parse_units, stages)
        if choices is None:
            return self._method_fallback(portion_result, "standard_portions", "standard_portion_contract", stages,
                catalog_eligible_leaf_count=len(eligible), catalog_retained_leaf_ids=retained, option_choices=[],
                standard_portion_source_sha256=source_hash)
        for choice in choices:
            masses[choice["component_id"]] = choice["grams"]
        return self._aggregate(identity, masses, deepcopy(portion_result.get("stages", [])) + stages,
            method="standard_portions", method_fallback=False, method_fallback_reason=None,
            catalog_eligible_leaf_count=len(eligible), catalog_retained_leaf_ids=retained, option_choices=choices,
            standard_portion_source_sha256=source_hash)
