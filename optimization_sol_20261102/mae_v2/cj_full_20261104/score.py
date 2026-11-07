"""Offline paired N5 scores; C uses its original shared-mass calibration rule."""
from __future__ import annotations

from copy import deepcopy
import csv
import importlib.util
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
sys.path.insert(0, str(V2 / "src"))
import opt_metrics
from opt_provider import atomic_json, fingerprint, sha
from common import load, usage_summary


def historical_mass_module():
    directory = V2 / "deliverables/j_c_20261104/C/source/optimization_sol_20261102/mae_v2/src"
    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        return value
    scoring = module("cj_c_scoring_as_run", directory / "opt_metrics.py")
    previous = sys.modules.get("opt_metrics")
    try:
        sys.modules["opt_metrics"] = scoring
        return module("cj_c_mass_as_run", directory / "metrics_v2.py")
    finally:
        sys.modules["opt_metrics"] = previous


def profile_stats(rows, labels, records, dataset="nutrition5k"):
    result = opt_metrics.evaluate(rows, labels, records)
    checks = [c for c in result["checks"] if c["dataset"] == dataset]
    values = {"n": len(records), "complete_groups": len({r["group_id"] for r in rows}),
        "metrics": result["datasets"][dataset]["primary"]["metrics"],
        "passed_checks": sum(c["pass"] for c in checks), "total_checks": 10,
        "worst_ratio": max(c["value"] / c["threshold"] if c["value"] is not None else float("inf") for c in checks),
        "fallback_n": sum(bool(r.get("fallback")) for r in records),
        "method_fallback_n": sum(bool(r.get("method_fallback")) for r in records),
        "repair_n": sum(bool(r.get("repair_used")) for r in records)}
    if dataset == "asa24":
        values["primary_scope"] = "conditional_Exact_only"
        values["full_dataset_statistics"] = result["datasets"][dataset]
    if not math.isfinite(values["worst_ratio"]):
        values["worst_ratio"] = None
    return values


def completed_records(rows, profile, frozen):
    records = load(HERE / "run" / profile / "records.json")
    expected = {r["sample_id"] for r in rows}
    count = 529 if profile == "ASA" else 314
    if len(records) != count or {r["sample_id"] for r in records} != expected:
        raise ValueError("complete_fixed_population_required_before_scoring")
    status = load(HERE / "run" / profile / "completion.json")
    if (status.get("profile") != profile or status.get("records") != count
            or status.get("freeze_sha256") != frozen
            or status["records_sha256"] != sha(HERE / "run" / profile / "records.json")):
        raise ValueError("completed_prediction_binding_changed")
    current = {r["sample_id"]: r for r in rows}
    if any(any(record.get(key) != current[record["sample_id"]][key]
               for key in ("dataset", "group_id", "fold")) for record in records):
        raise ValueError("completed_prediction_group_or_fold_changed")
    return records


def validate_conditional_records(rows, labels, records):
    for record in records:
        sid = record["sample_id"]
        exact = labels[sid]["food_code"] in record.get("final_codes", [])
        branch = record.get("conditional")
        if exact and (not isinstance(branch, dict) or branch.get("code") != labels[sid]["food_code"]
                      or branch.get("prediction") is None):
            raise ValueError("Exact_requires_explicit_conditional_prediction_or_fallback")
        if not exact and branch is not None:
            raise ValueError("nonExact_cannot_have_conditional_prediction")


def score_all(rows, labels, asa_catalog):
    n5_rows = [r for r in rows if r["dataset"] == "nutrition5k"]
    asa_rows = [r for r in rows if r["dataset"] == "asa24"]
    if len(n5_rows) != 314 or len(asa_rows) != 529:
        raise ValueError("joint_843_development_population_required")
    frozen = fingerprint(load(HERE / "freeze.json"))
    original = {p: completed_records(n5_rows, p, frozen) for p in ("C", "J")}
    asa_raw = completed_records(asa_rows, "ASA", frozen)
    validate_conditional_records(asa_rows, labels, asa_raw)
    seal = load(HERE / "run/ASA/identity_seal" / (fingerprint("all") + ".json"))
    if (seal["binding"]["freeze_sha256"] != frozen
            or set(seal["value"]) != {r["sample_id"] for r in asa_rows}
            or seal["value_sha256"] != fingerprint(seal["value"])):
        raise ValueError("all_529_blind_identities_must_be_sealed_before_scoring")
    for record in asa_raw:
        sid = record["sample_id"]
        identity = load(HERE / "run/ASA/identity" / (fingerprint(sid) + ".json"))
        if (identity["binding"]["freeze_sha256"] != frozen
                or fingerprint(identity["value"]) != identity["value_sha256"]
                or fingerprint(identity["value"]) != seal["value"][sid]
                or identity["value"].get("final_codes", []) != record.get("final_codes", [])):
            raise ValueError("ASA_identity_seal_or_record_codes_changed")
    mass = historical_mass_module()
    calibrated = mass.crossvalidate(n5_rows, labels, original["C"], calibrate=True, dataset="nutrition5k")
    atomic_json(HERE / "C_calibrated_oof.json", calibrated)
    asa_cal = mass.crossvalidate(asa_rows, labels, asa_raw, asa_catalog["asa_options"],
        {code: food["density"] for code, food in asa_catalog["foods"].items()}, calibrate=True, dataset="asa24")
    atomic_json(HERE / "ASA_calibrated_oof.json", asa_cal)
    stats = {"C_raw": profile_stats(n5_rows, labels, original["C"]),
             "C_calibrated": profile_stats(n5_rows, labels, calibrated["predictions"]),
             "J_raw": profile_stats(n5_rows, labels, original["J"]),
             "ASA_raw": profile_stats(asa_rows, labels, asa_raw, "asa24"),
             "ASA_calibrated": profile_stats(asa_rows, labels, asa_cal["predictions"], "asa24")}
    combinations = {}
    for name, n5, asa in (("C_raw", original["C"], asa_raw),
        ("C_retained", calibrated["predictions"], asa_cal["predictions"]),
        ("J_raw", original["J"], asa_raw), ("J_retained", original["J"], asa_cal["predictions"])):
        combined = opt_metrics.evaluate(rows, labels, n5 + asa)
        combined["MAE_is_primary_recognition_separately_reported"] = True
        combinations[name] = combined
    # These two complete profiles were fixed before running, not assembled from
    # whichever nutrition columns happen to look best on this development set.
    result = {"scope": "843_development_images_not_independent_acceptance", "profiles": stats,
        "predeclared_comparison": ["C_retained", "J_retained"], "combinations": combinations,
        "ASA_shared_once": True, "ASA_shared_calibrated_sha256": fingerprint(asa_cal["predictions"]),
        "C_raw_is_separately_reported": True, "J_new_calibration": False,
        "independent_acceptance_opened": False, "usage": [usage_summary(p) for p in ("C", "J", "ASA")]}
    atomic_json(HERE / "summary.json", result)
    with (HERE / "results.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["profile", "target", "mae", "per_unit", "mae_limit", "per_unit_limit", "pass_mae", "pass_per_unit"])
        for name, values in stats.items():
            for key, metric in values["metrics"].items():
                writer.writerow([name, key, metric["mae"], metric["per_unit"], metric["threshold_mae"],
                                 metric["threshold_per_unit"], metric["pass_mae"], metric["pass_per_unit"]])
    return result
