"""Pure v2 scoring, one-mass calibration, and bounded candidate selection.

This module never reads files or calls providers. Its caller supplies explicitly
development-only records for fitting. OOF model and calibration selection remain
development evidence; they are not independent acceptance results.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math
from typing import Mapping

import opt_metrics as scoring

KEYS = scoring.KEYS
THRESHOLDS = scoring.THRESHOLDS
DATASETS = scoring.DATASETS
evaluate = scoring.evaluate
median_option = scoring.median_option
nearest_option = scoring.nearest_option
option_prediction = scoring.option_prediction
fold_medians = scoring.fold_medians


def minimum_exact_n(n):
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("recognition population must be a nonnegative integer")
    return (15 * n + 99) // 100


def ratios(metrics, dataset):
    """The ten fixed MAE/threshold ratios; undefined metrics cannot pass."""
    if dataset not in DATASETS:
        raise ValueError("unknown dataset")
    rows = metrics["datasets"][dataset]["primary"]["metrics"]
    values = []
    for key in KEYS:
        for stat, threshold in zip(("mae", "per_unit"), THRESHOLDS[dataset][key]):
            value = rows[key][stat]
            values.append(math.inf if value is None else scoring._number(value, stat) / threshold)
    return values


def _objective(metrics, dataset):
    values = ratios(metrics, dataset)
    return max(values), sum(values) / len(values)


def select_candidate(candidates, dataset, registered_order=None):
    """Select a raw winner by (worst ratio, mean ratio, chats, registry order).

    Each candidate has name, metrics, estimated_chat_calls and registration_index
    (the last can instead be supplied by registered_order). ASA24 candidates must
    satisfy the 15% all-row inclusion floor. Returns None if none is eligible.
    The caller compares calibration only after this raw winner has been chosen.
    """
    return _select(candidates, dataset, registered_order, allow_calibrated=False)


def select_profile(candidates, dataset, registered_order=None):
    """Select final profiles after the bounded winner-only calibration checks.

    A retained calibration replaces its raw winner under the same registration;
    raw/calibrated versions must not be inserted as separate candidates. Entries
    with no scored population for this dataset are skipped. Names and registry
    positions remain unique across the entire supplied candidate collection.
    """
    return _select(candidates, dataset, registered_order, allow_calibrated=True)


def _select(candidates, dataset, registered_order, *, allow_calibrated):
    if dataset not in DATASETS:
        raise ValueError("unknown dataset")
    order = None
    if registered_order is not None:
        if len(registered_order) != len(set(registered_order)):
            raise ValueError("duplicate registered candidate")
        order = {name: i for i, name in enumerate(registered_order)}
    names, positions, eligible = set(), set(), []
    for candidate in candidates:
        name = candidate.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("candidate names must be distinct nonempty strings")
        names.add(name)
        if order is not None:
            if name not in order:
                raise ValueError("candidate absent from registered order")
            position = order[name]
        else:
            position = candidate.get("registration_index")
        if isinstance(position, bool) or not isinstance(position, int) or position < 0 or position in positions:
            raise ValueError("distinct nonnegative registered indices are required")
        positions.add(position)
        metrics = candidate["metrics"]
        data = metrics.get("datasets", {}).get(dataset)
        if data is None or data.get("n", 0) == 0:
            continue
        if not allow_calibrated and (candidate.get("calibrated") or candidate.get("calibration_enabled")):
            raise ValueError("select a raw winner before comparing calibration")
        calls = scoring._number(candidate.get("estimated_chat_calls"), "estimated_chat_calls")
        if dataset == "asa24":
            n, hits = data["n"], data["exact_n"]
            if isinstance(hits, bool) or not isinstance(hits, int) or not 0 <= hits <= n:
                raise ValueError("invalid Exact count")
            if not n or hits < minimum_exact_n(n):
                continue
        score = (*_objective(metrics, dataset), calls, position)
        eligible.append((score, candidate))
    if not eligible:
        return None
    score, winner = min(eligible, key=lambda item: item[0])
    result = deepcopy(winner)
    result["selection_score"] = dict(zip(
        ("worst_ratio", "mean_ratio", "estimated_chat_calls", "registration_index"),
        (value if math.isfinite(value) else None for value in score)))
    result["selection_scope"] = ("development_profile_selection_not_independent_validation" if allow_calibrated
                                 else "development_raw_selection_not_independent_validation")
    return result


def _training(manifest, labels, preds=None):
    scoring._training(manifest, labels)
    allowed = {"dev", "development", "train", "training"}
    for row in manifest:
        if not any(str(row.get(k, "")).lower() in allowed for k in ("split", "partition", "cohort")):
            raise ValueError("fitting requires explicit development provenance")
    return scoring._validate(manifest, labels, preds)


def _subset_labels(manifest, labels):
    return {row["sample_id"]: labels[row["sample_id"]] for row in manifest}


def _has_examples(preds, examples=False):
    return bool(examples or any(p.get("example_sources") or p.get("examples")
                               or (p.get("config") or {}).get("examples") for p in preds))


def _raw_leaves(pred):
    leaves = None
    if pred.get("raw_leaves") is not None:
        leaves = pred["raw_leaves"]
    else:
        portion = pred.get("portion_result") or {}
        for key in ("raw_leaves", "leaves", "portions"):
            if portion.get(key) is not None:
                leaves = portion[key]
                break
    if leaves is None:
        return None
    if not isinstance(leaves, list) or not leaves or any(not isinstance(leaf, Mapping) for leaf in leaves):
        raise ValueError("a complete nonempty leaf list is required")
    with_density = sum(leaf.get("density") is not None for leaf in leaves)
    if with_density == 0:
        # v1 portions identify estimates but did not store per-leaf densities.
        # Their five raw totals support the fixed-density algebraic equivalent.
        return None
    if with_density != len(leaves):
        raise ValueError("partially populated leaf densities cannot be calibrated")
    return leaves


def _raw_vector(pred):
    value = pred.get("raw_prediction")
    if value is None:
        value = (pred.get("portion_result") or {}).get("raw_prediction")
    if value is None:
        value = pred.get("prediction")
    return scoring._vector(value, "raw_prediction")


def _sum_leaves(leaves):
    if not isinstance(leaves, list) or not leaves:
        raise ValueError("a complete nonempty leaf list is required")
    totals = dict.fromkeys(KEYS, 0.0)
    identifiers, parents = set(), defaultdict(float)
    for leaf in leaves:
        identifier = leaf.get("component_id", leaf.get("item_id"))
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("leaf component identifiers must be unique")
        identifiers.add(identifier)
        grams = scoring._number(leaf.get("grams"), "leaf.grams")
        if grams <= 0:
            raise ValueError("every accounted leaf requires strictly positive grams")
        density = leaf.get("density")
        if not isinstance(density, Mapping):
            raise ValueError("every leaf requires its nutrient density")
        totals["weight"] += grams
        for key in KEYS[1:]:
            totals[key] += grams * scoring._number(density.get(key), f"density.{key}") / 100
        parent = leaf.get("parent_id")
        if parent is not None:
            if not isinstance(parent, str) or not parent:
                raise ValueError("invalid parent identifier")
            parents[parent] += grams
    return scoring._vector(totals, "leaf totals"), dict(parents)


def _close_vectors(actual, expected):
    return all(math.isclose(actual[k], expected[k], rel_tol=1e-9, abs_tol=1e-6) for k in KEYS)


def apply_mass(pred, factor):
    """Return a new N5 record, scaling leaves once and recomputing nutrients.

    For retained legacy predictions without leaves, scaling all five raw targets
    equally is the algebraically equivalent fixed-density representation. Raw
    artifacts remain unchanged; calibrated leaves/parents are separate evidence.
    """
    factor = scoring._number(factor, "mass_factor")
    if not .9 <= factor <= 1.125:
        raise ValueError("the final shrunk mass factor must be in [.9,1.125]")
    if pred.get("dataset") != "nutrition5k":
        raise ValueError("apply_mass is Nutrition5k only")
    result = deepcopy(pred)
    if pred.get("fallback", False):
        result["mass_factor"] = 1.0
        return result
    raw = _raw_vector(pred)
    leaves = _raw_leaves(pred)
    if leaves is not None:
        totals, parents = _sum_leaves(leaves)
        if not _close_vectors(totals, raw):
            raise ValueError("raw leaf mass/nutrients disagree with raw_prediction")
        recorded_parents = (pred.get("portion_result") or {}).get("parents")
        if recorded_parents is not None:
            if (not isinstance(recorded_parents, list)
                    or len(recorded_parents) != len(parents)
                    or {p.get("parent_id") for p in recorded_parents} != set(parents)):
                raise ValueError("raw parent aggregates disagree with leaves")
            for parent in recorded_parents:
                grams = scoring._number(parent.get("grams"), "parent.grams")
                if not math.isclose(grams, parents[parent["parent_id"]], rel_tol=1e-9, abs_tol=1e-6):
                    raise ValueError("raw parent mass disagrees with its leaves")
        calibrated = deepcopy(leaves)
        for leaf in calibrated:
            leaf["grams"] = scoring._number(leaf["grams"], "leaf.grams") * factor
        vector, parent_totals = _sum_leaves(calibrated)
        result["raw_leaves"] = deepcopy(leaves)
        result["calibrated_leaves"] = calibrated
        result["calibrated_parents"] = [{"parent_id": key, "grams": value}
                                        for key, value in sorted(parent_totals.items())]
        result["mass_calibration_representation"] = "leaf_mass_with_fixed_density"
    else:
        vector = {key: raw[key] * factor for key in KEYS}
        result["mass_calibration_representation"] = "equivalent_raw_vector_scaling"
    result["raw_prediction"] = dict(raw)
    result["prediction"] = scoring._vector(vector, "mass calibrated prediction")
    result["mass_factor"] = factor
    return result


def fit_mass(manifest, labels, preds, dataset, options_by_code=None, nutrients_by_code=None, *, examples=False):
    """Fit one unshrunk .800..1.250 grid factor; return its half-shrunk value."""
    if dataset not in DATASETS:
        raise ValueError("unknown dataset")
    mi, pi = _training(manifest, labels, preds)
    if _has_examples(preds, examples):
        raise ValueError("supervised calibration of few-shot references is forbidden")
    eligible = []
    for sid, row in mi.items():
        if row["dataset"] != dataset:
            continue
        pred, label = pi[sid], labels[sid]
        if dataset == "nutrition5k":
            if pred.get("fallback", False):
                continue
            raw = _raw_vector(pred)
            # Check available leaf accounting once, outside the fitting grid.
            apply_mass(pred, 1.0)
            if raw["weight"] > 0:
                eligible.append((row, label, raw))
        else:
            branch = pred.get("conditional") or {}
            if (label["food_code"] in pred.get("final_codes", [])
                    and not branch.get("fallback", False) and branch.get("prediction") is not None
                    and branch.get("continuous_grams") is not None):
                code = label["food_code"]
                if not options_by_code or code not in options_by_code or not nutrients_by_code or code not in nutrients_by_code:
                    raise ValueError("Exact calibration requires the complete option catalog and density")
                scoring._options(options_by_code[code])
                eligible.append((row, label, branch))
    groups = {row["group_id"] for row, _, _ in eligible}
    info = {"dataset": dataset, "eligible_n": len(eligible), "eligible_groups": len(groups),
            "mass_factor": 1.0, "raw_factor": 1.0, "grid": {"minimum": .8, "maximum": 1.25, "step": .001},
            "shrinkage_toward_one": .5, "training_ids": [r["sample_id"] for r in manifest],
            "eligible_ids": [r["sample_id"] for r, _, _ in eligible]}
    if len(groups) < 5:
        info["reason"] = "fewer_than_five_eligible_training_groups"
        return info
    best = None
    for integer in range(800, 1251):
        factor = integer / 1000
        values = []
        for _, label, raw in eligible:
            if dataset == "nutrition5k":
                values.append({key: raw[key] * factor for key in KEYS})
            else:
                code = label["food_code"]
                option = nearest_option(raw["continuous_grams"] * factor, options_by_code[code])
                values.append(option_prediction(option, nutrients_by_code[code]))
        stats = scoring._stats([label for _, label, _ in eligible], values, dataset)
        q = ratios({"datasets": {dataset: {"primary": stats}}}, dataset)
        score = (max(q), sum(q) / len(q), abs(integer - 1000), integer)
        if best is None or score < best:
            best = score
    raw_factor = best[-1] / 1000
    info.update(raw_factor=raw_factor, mass_factor=1 + .5 * (raw_factor - 1), reason="fitted",
                training_objective={"worst_ratio": best[0] if math.isfinite(best[0]) else None,
                                    "mean_ratio": best[1] if math.isfinite(best[1]) else None})
    return info


def fit_final(manifest, labels, preds, options_by_code=None, nutrients_by_code=None, *, dataset,
              enabled=True, examples=False):
    """Fit final parameters on the development population only, never validation."""
    _training(manifest, labels, preds)
    rows = [r for r in manifest if r["dataset"] == dataset]
    if not rows:
        raise ValueError("no training records for requested dataset")
    ids = {r["sample_id"] for r in rows}
    subset = [p for p in preds if p["sample_id"] in ids]
    scoped_labels = _subset_labels(rows, labels)
    if enabled:
        params = fit_mass(rows, scoped_labels, subset, dataset, options_by_code, nutrients_by_code, examples=examples)
    else:
        params = {"dataset": dataset, "mass_factor": 1.0, "raw_factor": 1.0,
                  "reason": "calibration_disabled", "training_ids": [r["sample_id"] for r in rows]}
    params.update(enabled=bool(enabled), fallback=fold_medians(rows, scoped_labels, dataset=dataset))
    return params


def apply_parameters(pred, parameters, options_by_code=None, nutrients_by_code=None):
    """Apply frozen parameters without receiving any target labels.

    ASA24's explicit conditional adapter already supplies branch.code. Only its
    continuous estimate is scaled; the final prediction always is a legal option.
    """
    dataset = parameters["dataset"]
    if pred.get("dataset") != dataset:
        raise ValueError("parameter and prediction datasets differ")
    factor = parameters["mass_factor"]
    if not .9 <= scoring._number(factor, "mass_factor") <= 1.125:
        raise ValueError("invalid final mass factor")
    result = deepcopy(pred)
    if dataset == "nutrition5k":
        if result.get("fallback", False):
            result.update(prediction=dict(parameters["fallback"]), fallback=True, native_complete=False,
                          mass_factor=1.0, fallback_reason=result.get("fallback_reason") or "incomplete_prediction")
            return result
        return apply_mass(result, factor)
    if dataset != "asa24":
        raise ValueError("unknown dataset")
    e2e = result.get("e2e")
    if e2e and e2e.get("fallback") and e2e.get("option_id") is None:
        e2e["prediction"] = dict(parameters["fallback"])
        result["prediction"] = dict(parameters["fallback"])
    branch = result.get("conditional")
    if branch is None:
        return result
    code = branch.get("code")
    if not options_by_code or code not in options_by_code or not nutrients_by_code or code not in nutrients_by_code:
        raise ValueError("conditional branch requires its catalog code and density")
    if branch.get("prediction") is None or branch.get("fallback", False):
        option = median_option(options_by_code[code])
        branch.update(fallback=True, fallback_reason=branch.get("fallback_reason") or "incomplete_portion")
    elif branch.get("continuous_grams") is not None:
        option = nearest_option(branch["continuous_grams"] * factor, options_by_code[code])
    else:
        legal = {str(o["option_id"]): o for o in scoring._options(options_by_code[code])}
        if str(branch.get("option_id")) not in legal:
            raise ValueError("direct portion refers to an illegal option")
        option = legal[str(branch["option_id"])]
    branch.update(option_id=option["option_id"], prediction=option_prediction(option, nutrients_by_code[code]),
                  calibration_factor=factor)
    result["conditional"] = branch
    return result


def crossvalidate(manifest, labels, preds, options_by_code=None, nutrients_by_code=None, *,
                  calibrate=False, dataset=None, examples=False):
    """Fold-exclusive fitting and fallback, with explicit parameter provenance."""
    mi, pi = _training(manifest, labels, preds)
    if dataset is not None and dataset not in DATASETS:
        raise ValueError("unknown dataset")
    if any(row.get("fold") is None for row in manifest):
        raise ValueError("every development record requires a fold")
    if calibrate and _has_examples(preds, examples):
        raise ValueError("supervised calibration of few-shot references is forbidden")
    output = deepcopy(list(preds))
    index = scoring._index(output, "output")
    parameters = []
    for ds in DATASETS:
        rows = [r for r in manifest if r["dataset"] == ds]
        if not rows:
            continue
        folds = sorted({r["fold"] for r in rows}, key=str)
        if len(folds) < 2:
            raise ValueError("at least two nonempty group folds are required")
        enabled = bool(calibrate and (dataset is None or dataset == ds))
        for fold in folds:
            train = [r for r in rows if r["fold"] != fold]
            holdout = [r for r in rows if r["fold"] == fold]
            params = fit_final(train, _subset_labels(train, labels), [pi[r["sample_id"]] for r in train],
                               options_by_code, nutrients_by_code, dataset=ds, enabled=enabled, examples=examples)
            parameters.append({**deepcopy(params), "fold": fold,
                               "heldout_ids": [r["sample_id"] for r in holdout],
                               "training_groups": sorted({r["group_id"] for r in train}),
                               "heldout_groups": sorted({r["group_id"] for r in holdout})})
            for row in holdout:
                sid = row["sample_id"]
                pred = deepcopy(index[sid])
                if ds == "asa24":
                    # This is the already registered conditional adapter only.
                    code = labels[sid]["food_code"]
                    if code in pred.get("final_codes", []):
                        branch = pred.get("conditional") or {}
                        if branch.get("code") not in (None, code):
                            raise ValueError("conditional code disagrees with Exact reference code")
                        branch["code"] = code
                        pred["conditional"] = branch
                updated = apply_parameters(pred, params, options_by_code, nutrients_by_code)
                index[sid].clear()
                index[sid].update(updated)
    return {"predictions": output, "fold_parameters": parameters,
            "metrics": evaluate(manifest, labels, output),
            "scope": "development_oof_not_independent_validation"}


def calibration_improves(raw_metrics, calibrated_metrics, dataset):
    old, new = ratios(raw_metrics, dataset), ratios(calibrated_metrics, dataset)
    return max(new) < max(old) and not any(a <= 1 < b for a, b in zip(old, new))


def compare_calibration(manifest, labels, preds, options_by_code=None, nutrients_by_code=None, *,
                        dataset, examples=False):
    """Compare raw vs calibrated once for the caller's already selected raw arm."""
    raw = crossvalidate(manifest, labels, preds, options_by_code, nutrients_by_code, calibrate=False, examples=examples)
    forbidden = _has_examples(preds, examples)
    calibrated = None if forbidden else crossvalidate(manifest, labels, preds, options_by_code, nutrients_by_code,
                                                     calibrate=True, dataset=dataset)
    enabled = calibrated is not None and calibration_improves(raw["metrics"], calibrated["metrics"], dataset)
    selected = deepcopy(calibrated if enabled else raw)
    selected.update(raw_metrics=raw["metrics"], calibrated_metrics=calibrated["metrics"] if calibrated else None,
                    calibration_enabled=enabled,
                    calibration_reason="fewshot_calibration_forbidden" if forbidden else
                    ("strict_worst_improvement_without_lost_pass" if enabled else "calibration_not_retained"),
                    selected_parameters=fit_final(manifest, labels, preds, options_by_code, nutrients_by_code,
                                                  dataset=dataset, enabled=enabled, examples=examples))
    return selected
