#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text())


def rel_err(pred, truth):
    if truth is None or pred is None:
        return None
    if truth == 0:
        return None
    return abs(pred - truth) / abs(truth)


def main():
    parser = argparse.ArgumentParser(description="Score P27 out-of-sample predictions.")
    parser.add_argument("--cases", required=True, help="Comma-separated case IDs.")
    parser.add_argument("--pred-suffix", default="", help="Suffix for prediction file (e.g. _p28).")
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    rows = []
    max_err = {"e_load": 0.0, "u2": 0.0, "rho": 0.0}

    for case in cases:
        base = Path("outputs") / case / "analysis"
        pred_name = f"p27_pred{args.pred_suffix}.json"
        pred = load_json(base / pred_name)
        metrics = load_json(base / "metrics.json")
        metrics_u2 = load_json(base / "metrics_u2hist.json")
        metrics_form = load_json(base / "metrics_formation.json")

        truth_eload = metrics.get("e_load_J")
        truth_u2 = metrics_u2.get("u2_p99_at_stepOff")
        truth_rho = metrics_form.get("formation_kpi_phase", {}).get("rho_delta_rel_phase")

        err_eload = rel_err(pred.get("e_load_pred"), truth_eload)
        err_u2 = rel_err(pred.get("u2_p99_pred"), truth_u2)
        rho_pred_val = pred.get("rho_delta_rel_phase_pred")
        err_rho = rel_err(rho_pred_val, truth_rho)

        rho_truth_ge = truth_rho is not None and truth_rho >= 0.05
        rho_pred_ge = pred.get("rho_pred_ge_0p05")
        if rho_pred_ge is None:
            rho_pred_ge = rho_pred_val is not None and rho_pred_val >= 0.05
        rho_class_correct = None
        if truth_rho is not None and rho_pred_ge is not None:
            rho_class_correct = (rho_truth_ge == rho_pred_ge)
        err_rho_when_ge = err_rho if rho_truth_ge and rho_pred_val is not None else None

        for key, val in [("e_load", err_eload), ("u2", err_u2), ("rho", err_rho)]:
            if val is not None:
                max_err[key] = max(max_err[key], val)

        rows.append({
            "case": case,
            "e_load_pred": pred.get("e_load_pred"),
            "e_load_true": truth_eload,
            "rel_err_eload": err_eload,
            "u2_p99_pred": pred.get("u2_p99_pred"),
            "u2_p99_true": truth_u2,
            "rel_err_u2": err_u2,
            "rho_pred": rho_pred_val,
            "rho_true": truth_rho,
            "rel_err_rho": err_rho,
            "rho_truth_ge_0p05": rho_truth_ge,
            "rho_pred_ge_0p05": rho_pred_ge,
            "rho_class_correct": rho_class_correct,
            "rel_err_rho_when_truth_ge_0p05": err_rho_when_ge,
        })

    out = {
        "cases": rows,
        "max_rel_err": max_err,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
