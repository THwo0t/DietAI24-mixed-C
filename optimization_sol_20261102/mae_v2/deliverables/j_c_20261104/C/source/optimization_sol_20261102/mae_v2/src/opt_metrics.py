"""Pure, offline scoring and development-only calibration for the Sol study.

Public vectors use weight (g), energy (kcal), fat/carbs/protein (g). Nutrient
density vectors omit weight and are per 100 g. No function reads labels from
disk. Calibration callers must pass training manifests, never a sealed split.
``evaluate`` accepts either or both datasets; absent datasets cannot pass the
joint 20-metric acceptance test. Missing prediction records are errors, not
silently excluded observations. A missing ASA24 conditional branch remains an
Exact observation and makes the conditional population unscorable.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math
import random
import re
from statistics import median
from typing import Any, Mapping, Sequence

KEYS = ("weight", "energy", "fat", "carbs", "protein")
DATASETS = ("asa24", "nutrition5k")
THRESHOLDS = {
    "asa24": dict(zip(KEYS, ((64.22, .4524), (62.01, .2548), (2.34, .2262),
                            (9.035, .2704), (1.794, .3029)))),
    "nutrition5k": dict(zip(KEYS, ((58.63, .2028), (88.66, .2561), (5.213, .2938),
                                  (7.774, .2756), (5.044, .2444)))),
}
CODE = re.compile(r"[0-9]{8}\Z")


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context}: expected a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{context}: expected a finite nonnegative number")
    return result


def _vector(value: Any, context: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or any(k not in value for k in KEYS):
        raise ValueError(f"{context}: all five targets are required")
    return {k: _number(value[k], f"{context}.{k}") for k in KEYS}


def _code(value: Any) -> bool:
    return isinstance(value, str) and CODE.fullmatch(value) is not None


def _index(rows: Sequence[Mapping], context: str) -> dict[str, Mapping]:
    indexed = {}
    for row in rows:
        sid = row.get("sample_id")
        if not isinstance(sid, str) or not sid:
            raise ValueError(f"{context}: sample_id must be a nonempty string")
        if sid in indexed:
            raise ValueError(f"{context}: duplicate sample_id {sid}")
        indexed[sid] = row
    return indexed


def _validate(manifest, labels, preds=None):
    mi = _index(manifest, "manifest")
    pi = _index(preds, "predictions") if preds is not None else None
    if pi is not None and set(mi) != set(pi):
        raise ValueError("manifest and prediction sample IDs must match exactly")
    if not isinstance(labels, Mapping) or not set(mi).issubset(labels):
        raise ValueError("labels missing manifest sample IDs")
    for sid, row in mi.items():
        dataset = row.get("dataset")
        if dataset not in DATASETS:
            raise ValueError(f"{sid}: unknown dataset")
        if not isinstance(row.get("group_id"), str) or not row["group_id"]:
            raise ValueError(f"{sid}: nonempty group_id required")
        _vector(labels[sid].get("truth"), f"{sid}.truth")
        if dataset == "asa24" and not _code(labels[sid].get("food_code")):
            raise ValueError(f"{sid}: ASA24 food_code must be eight digits")
        if pi is None:
            continue
        pred = pi[sid]
        if pred.get("dataset") != dataset:
            raise ValueError(f"{sid}: prediction dataset mismatch")
        codes = pred.get("final_codes", [])
        if not isinstance(codes, list) or any(not _code(c) for c in codes):
            raise ValueError(f"{sid}: invalid final_codes")
        if len(codes) != len(set(codes)) or (dataset == "asa24" and len(codes) > 3):
            raise ValueError(f"{sid}: duplicate or excessive final_codes")
        primary = pred.get("primary_code")
        if (codes and primary != codes[0]) or (not codes and primary not in (None, "")):
            raise ValueError(f"{sid}: primary_code must identify the first final code")
        if "retrieved_codes" in pred:
            retrieved = pred["retrieved_codes"]
            if not isinstance(retrieved, list) or any(not _code(c) for c in retrieved):
                raise ValueError(f"{sid}: invalid retrieved_codes")
            if not set(codes).issubset(retrieved):
                raise ValueError(f"{sid}: selected code absent from retrieved candidates")
        for field in ("prediction", "raw_prediction"):
            if pred.get(field) is not None:
                _vector(pred[field], f"{sid}.{field}")
        if dataset == "nutrition5k" and pred.get("prediction") is None:
            raise ValueError(f"{sid}: Nutrition5k requires a scored prediction or fallback")
        for branch in ("conditional", "e2e"):
            value = pred.get(branch)
            if value is not None:
                if not isinstance(value, Mapping):
                    raise ValueError(f"{sid}.{branch}: expected an object")
                if value.get("prediction") is not None:
                    _vector(value["prediction"], f"{sid}.{branch}.prediction")
                if value.get("continuous_grams") is not None:
                    _number(value["continuous_grams"], f"{sid}.{branch}.continuous_grams")
        for field in ("native_complete", "fallback"):
            if field in pred and not isinstance(pred[field], bool):
                raise ValueError(f"{sid}.{field}: expected a boolean")
    return mi, pi


def _training(manifest, labels):
    """Fail closed on explicit non-development provenance and group fold leaks."""
    _validate(manifest, labels)
    seen = {}
    for row in manifest:
        for source in (row, labels[row["sample_id"]]):
            for field in ("split", "partition", "cohort"):
                marker = source.get(field)
                if marker is not None and str(marker).lower() not in {
                    "dev", "development", "train", "training"}:
                    raise ValueError("calibration only accepts development/training records")
            if source.get("sealed") or source.get("validation") or source.get("is_validation"):
                raise ValueError("sealed/validation records cannot be used for fitting")
        group = (row["dataset"], row["group_id"])
        fold = row.get("fold")
        if group in seen and seen[group] != fold:
            raise ValueError("one independent group occurs in multiple folds")
        seen[group] = fold


def _stats(rows, values, dataset):
    """All targets use exactly the same rows; any missing vector invalidates all."""
    n = len(rows)
    complete = n > 0 and all(v is not None for v in values)
    out = {"n": n, "available_n": sum(v is not None for v in values),
           "complete_population": complete, "metrics": {}}
    for key in KEYS:
        ref = sum(row["truth"][key] / n for row in rows) if n else None
        mae = (sum(abs(v[key] - row["truth"][key]) / n for row, v in zip(rows, values))
               if complete else None)
        per_unit = mae / ref if mae is not None and ref and ref > 0 else None
        if per_unit is not None and not math.isfinite(per_unit):
            per_unit = None
        absolute, relative = THRESHOLDS[dataset][key]
        out["metrics"][key] = {
            "mae": mae, "per_unit": per_unit, "reference_mean": ref,
            "threshold_mae": absolute, "threshold_per_unit": relative,
            "pass_mae": mae is not None and mae <= absolute,
            "pass_per_unit": per_unit is not None and per_unit <= relative,
        }
    return out


def _dataset_stats(rows, dataset):
    n = len(rows)
    labels = [r["label"] for r in rows]
    predictions = [r["pred"] for r in rows]
    base = {"n": n, "groups": len({r["manifest"]["group_id"] for r in rows}),
            "native_complete_n": sum(bool(p.get("native_complete")) and not p.get("fallback", False)
                                     and not p.get("method_fallback", False) for p in predictions),
            "complete_n": sum(p.get("prediction") is not None and not p.get("fallback", False)
                              for p in predictions),
            "method_fallback_n": sum(bool(p.get("method_fallback")) for p in predictions),
            "fallback_n": sum(bool(p.get("fallback")) for p in predictions),
            "repair_success_n": sum(bool(p.get("repair_used") or p.get("repaired"))
                                    and not p.get("fallback", False) for p in predictions)}
    for field in ("native_complete", "complete", "fallback", "repair_success", "method_fallback"):
        base[f"{field}_rate"] = base[f"{field}_n"] / n if n else None
    base["enhancement_native_n"] = base["native_complete_n"]
    base["enhancement_native_rate"] = base["native_complete_rate"]
    if dataset == "nutrition5k":
        base["primary"] = _stats(labels, [p["prediction"] for p in predictions], dataset)
        native = [r for r in rows if r["pred"].get("native_complete")
                  and not r["pred"].get("fallback", False) and not r["pred"].get("method_fallback", False)]
        base["native_subset"] = _stats([r["label"] for r in native],
                                      [r["pred"]["prediction"] for r in native], dataset)
        complete = [r for r in rows if r["pred"].get("prediction") is not None
                    and not r["pred"].get("fallback", False)]
        base["complete_subset"] = _stats([r["label"] for r in complete],
                                        [r["pred"]["prediction"] for r in complete], dataset)
        return base
    exact = [r for r in rows if r["label"]["food_code"] in r["pred"].get("final_codes", [])]
    first = sum(p.get("primary_code") == label["food_code"] for p, label in zip(predictions, labels))
    coded = sum(bool(p.get("final_codes")) for p in predictions)
    recalled = sum(label["food_code"] in p.get("retrieved_codes", [])
                   for p, label in zip(predictions, labels))
    required = (15 * n + 99) // 100  # v2: exact integer ceil(0.15*n)
    base.update({"exact_n": len(exact), "exact_rate": len(exact) / n if n else None,
                 "first_n": first, "first_rate": first / n if n else None,
                 "coded_n": coded, "exact_among_coded": len(exact) / coded if coded else None,
                 "retrieval_exact_n": recalled, "retrieval_exact_rate": recalled / n if n else None,
                 "minimum_exact_n": required, "recognition_pass": bool(n and len(exact) >= required),
                 "average_final_code_count": sum(len(p.get("final_codes", [])) for p in predictions) / n if n else None,
                 "exact_groups": len({r["manifest"]["group_id"] for r in exact}), "by_size": {}})
    for size in sorted({str(r["label"].get("size", "unknown")) for r in rows}):
        sized = [r for r in rows if str(r["label"].get("size", "unknown")) == size]
        sized_exact = [r for r in sized if r["label"]["food_code"] in r["pred"].get("final_codes", [])]
        sized_first = sum(r["pred"].get("primary_code") == r["label"]["food_code"] for r in sized)
        base["by_size"][size] = {
            "n": len(sized), "exact_n": len(sized_exact), "exact_rate": len(sized_exact) / len(sized),
            "first_n": sized_first, "first_rate": sized_first / len(sized),
            "conditional": _stats([r["label"] for r in sized_exact],
                                  [(r["pred"].get("conditional") or {}).get("prediction") for r in sized_exact], dataset),
            "e2e": _stats([r["label"] for r in sized],
                          [(r["pred"].get("e2e") or {}).get("prediction") for r in sized], dataset)}
    conditional = [r["pred"].get("conditional") or {} for r in exact]
    base["primary"] = _stats([r["label"] for r in exact], [p.get("prediction") for p in conditional], dataset)
    base["conditional_fallback_n"] = sum(bool(p.get("fallback")) for p in conditional)
    base["conditional_missing_n"] = sum(p.get("prediction") is None for p in conditional)
    base["conditional_fallback_rate"] = base["conditional_fallback_n"] / len(exact) if exact else None
    base["e2e"] = _stats(labels, [(p.get("e2e") or {}).get("prediction") for p in predictions], dataset)
    base["e2e_fallback_n"] = sum(bool((p.get("e2e") or {}).get("fallback")) for p in predictions)
    return base


def _quantile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def _bootstrap(rows, dataset, count, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row["manifest"]["group_id"]].append(row)
    names = sorted(groups)
    samples = defaultdict(list)
    empty_exact = 0
    rng = random.Random(seed)
    for _ in range(count):
        selected = [r for name in rng.choices(names, k=len(names)) for r in groups[name]] if names else []
        score = _dataset_stats(selected, dataset)
        if dataset == "asa24":
            empty_exact += score["exact_n"] == 0
            if score["exact_rate"] is not None:
                samples["exact_rate"].append(score["exact_rate"])
            if score["first_rate"] is not None:
                samples["first_rate"].append(score["first_rate"])
        for key in KEYS:
            for statistic in ("mae", "per_unit"):
                value = score["primary"]["metrics"][key][statistic]
                if value is not None:
                    samples[f"{key}.{statistic}"].append(value)
    intervals = {key: {"low": _quantile(values, .025), "high": _quantile(values, .975),
                       "valid_replicates": len(values)} for key, values in samples.items()}
    # Keep undefined intervals explicit, including an empty Exact population.
    for key in KEYS:
        for statistic in ("mae", "per_unit"):
            intervals.setdefault(f"{key}.{statistic}", {"low": None, "high": None, "valid_replicates": 0})
    return {"replicates": count, "seed": seed, "unit": "manifest_group_id",
            "group_count": len(names),
            "confidence": .95, "intervals": intervals,
            "empty_exact_replicates": empty_exact if dataset == "asa24" else None,
            "empty_exact_fraction": empty_exact / count if dataset == "asa24" else None}


def evaluate(manifest, labels, preds, bootstrap=0, seed=20261102):
    """Score fixed manifest populations; ``bootstrap=True`` means 10000 draws.

    Returns ``datasets``, 20 ``checks``, ``passed_checks``, ``all_mae_pass``,
    ``recognition_pass`` and ``joint_pass``. Null metrics are explicit failures.
    The denominator of ASA24 recognition is always the full ASA24 manifest.
    """
    mi, pi = _validate(manifest, labels, preds)
    if isinstance(bootstrap, bool):
        bootstrap = 10000 if bootstrap else 0
    if not isinstance(bootstrap, int) or bootstrap < 0:
        raise ValueError("bootstrap must be a nonnegative integer or boolean")
    datasets, checks = {}, []
    for dataset in DATASETS:
        rows = [{"manifest": row, "label": labels[sid], "pred": pi[sid]}
                for sid, row in mi.items() if row["dataset"] == dataset]
        score = _dataset_stats(rows, dataset)
        if bootstrap:
            score["bootstrap"] = _bootstrap(rows, dataset, bootstrap, seed)
        datasets[dataset] = score
        for key in KEYS:
            metric = score["primary"]["metrics"][key]
            for statistic in ("mae", "per_unit"):
                checks.append({"dataset": dataset, "target": key, "statistic": statistic,
                               "value": metric[statistic], "threshold": metric[f"threshold_{statistic}"],
                               "pass": metric[f"pass_{statistic}"]})
    passed = sum(c["pass"] for c in checks)
    recognition = datasets["asa24"]["recognition_pass"]
    return {"version": 2, "datasets": datasets, "checks": checks,
            "passed_checks": passed, "total_checks": 20, "all_mae_pass": passed == 20,
            "recognition_pass": recognition, "joint_pass": passed == 20 and recognition}


def fit_n5(manifest, labels, preds):
    """Compatibility entrypoint: v2 always uses one shared mass factor."""
    from metrics_v2 import fit_mass
    factor = fit_mass(manifest, labels, preds, "nutrition5k")["mass_factor"]
    return dict.fromkeys(KEYS, factor)


def apply_n5(prediction, factors):
    """Reject independent nutrient calibration; only shared mass scaling exists."""
    prediction = _vector(prediction, "prediction")
    factors = _vector(factors, "factors")
    if len(set(factors.values())) != 1 or not .9 <= factors["weight"] <= 1.125:
        raise ValueError("v2 requires one shared shrunk mass factor in [.9,1.125]")
    return _vector({k: prediction[k] * factors[k] for k in KEYS}, "calibrated")


def fold_medians(manifest, labels, heldout_fold=None, dataset="nutrition5k"):
    """Training-only component medians, excluding ``heldout_fold`` if supplied."""
    _training(manifest, labels)
    selected = [r for r in manifest if r["dataset"] == dataset
                and (heldout_fold is None or r.get("fold") != heldout_fold)]
    if not selected:
        raise ValueError("cannot fit fallback medians without training records")
    return {k: float(median(labels[r["sample_id"]]["truth"][k] for r in selected)) for k in KEYS}


def _options(options):
    if not isinstance(options, (list, tuple)) or not options:
        raise ValueError("a nonempty legal option catalog is required")
    ids = set()
    for option in options:
        if not isinstance(option, Mapping) or option.get("option_id") is None:
            raise ValueError("legal option requires option_id")
        identifier = str(option["option_id"])
        if identifier in ids:
            raise ValueError("duplicate option_id")
        ids.add(identifier)
        _number(option.get("grams"), "option.grams")
    return list(options)


def nearest_option(grams, options):
    """Nearest legal weight; deterministic ties prefer lower grams then ID."""
    grams = _number(grams, "grams")
    return deepcopy(min(_options(options), key=lambda option:
                        (abs(option["grams"] - grams), option["grams"], str(option["option_id"]))))


def median_option(options):
    """Legal option nearest the catalog median, with the same frozen tie rule."""
    options = _options(options)
    return nearest_option(median(option["grams"] for option in options), options)


def option_prediction(option, nutrients_per_100g):
    """Convert a legal option to five targets using FNDDS per-100 g density."""
    grams = _number(option.get("grams"), "option.grams")
    nutrients = {k: _number(nutrients_per_100g.get(k), f"density.{k}") for k in KEYS[1:]}
    return _vector({"weight": grams, **{k: grams * nutrients[k] / 100 for k in KEYS[1:]}}, "option prediction")


def fit_asa(manifest, labels, preds, options_by_code, nutrients_by_code):
    """Compatibility entrypoint for the v2 grid and shrinkage policy."""
    from metrics_v2 import fit_mass
    return fit_mass(manifest, labels, preds, "asa24", options_by_code, nutrients_by_code)["mass_factor"]


def crossvalidate(manifest, labels, preds, options_by_code=None, nutrients_by_code=None,
                  calibrate_n5=True, calibrate_asa=True):
    """Compatibility adapter; new drivers should import metrics_v2 directly."""
    from metrics_v2 import crossvalidate as v2_crossvalidate
    dataset = None if calibrate_n5 == calibrate_asa else ("nutrition5k" if calibrate_n5 else "asa24")
    return v2_crossvalidate(manifest, labels, preds, options_by_code, nutrients_by_code,
                            calibrate=calibrate_n5 or calibrate_asa, dataset=dataset)
