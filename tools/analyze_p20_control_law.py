#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def load_json(path: Path):
    return json.loads(path.read_text())


def get_case_values(case_id: str):
    base = Path("outputs") / case_id / "analysis"
    metrics = load_json(base / "metrics.json")
    metrics_u2 = load_json(base / "metrics_u2hist.json")
    metrics_form = load_json(base / "metrics_formation.json")
    args_path = Path("cases") / case_id / "inputs" / "warpx_args.json"
    args = load_json(args_path)

    drive_amp = args.get("driveAmp_scale")
    repeat = args.get("inject_repeat_nsteps")
    off_step = args.get("drive_envelope_off_step")
    repeat_pre = None
    if off_step is not None and repeat is not None:
        repeat_pre = min(repeat, off_step)
    u2_p99 = metrics_u2.get("u2_p99_at_stepOff")
    e_load = metrics.get("e_load_J")
    rho_phase = metrics_form.get("formation_kpi_phase", {}).get("rho_delta_rel_phase")

    return {
        "case": case_id,
        "driveAmp_scale": drive_amp,
        "inject_repeat_nsteps": repeat,
        "u2_p99_at_stepOff": u2_p99,
        "e_load": e_load,
        "rho_delta_rel_phase": rho_phase,
        "off_step": off_step,
        "repeat_pre": repeat_pre,
    }


def fit_log_linear(y, X, labels):
    y = np.array(y, dtype=float)
    X = np.array(X, dtype=float)
    Y = np.log(y)
    Xmat = np.column_stack([np.ones(len(X)), X])
    coeffs, _, _, _ = np.linalg.lstsq(Xmat, Y, rcond=None)
    yhat = Xmat @ coeffs
    sst = np.sum((Y - np.mean(Y)) ** 2)
    sse = np.sum((Y - yhat) ** 2)
    r2 = None if sst == 0 else 1.0 - sse / sst
    return {
        "coeffs": {"intercept": float(coeffs[0]), **{labels[i]: float(coeffs[i + 1]) for i in range(len(labels))}},
        "r2": None if r2 is None else float(r2),
    }


def main():
    parser = argparse.ArgumentParser(description="Fit simple control-law proxies for P20.")
    parser.add_argument("--cases", nargs="+", required=True, help="Case IDs to include in fit.")
    parser.add_argument("--rho-model", choices=["loglin", "logquad", "logquad2d"], default="loglin")
    parser.add_argument("--tail-model", choices=["loglin", "logquad", "logquad2d", "piecewise_Lpre"], default="loglin")
    parser.add_argument("--tail-knot", type=float, default=256.0, help="Repeat knot for piecewise tail model (in repeat units).")
    parser.add_argument("--repeat-pre", action="store_true", help="Use repeat_pre=min(repeat, off_step) for tail/rho.")
    parser.add_argument("--rho-strong-clf", choices=["none", "logistic"], default="none")
    parser.add_argument("--rho-strong-threshold", type=float, default=0.05, help="Threshold for strong compression classification.")
    parser.add_argument(
        "--rho-strong-features",
        choices=["basic", "quad"],
        default="quad",
        help="Feature set for rho strong classifier: basic=[1,Lpre,Ld,Lpre^2], quad adds Ld^2 and Lpre*Ld.",
    )
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument("--csv", help="Optional CSV output path.")
    args = parser.parse_args()

    rows = [get_case_values(c) for c in args.cases]

    # Tail fit: log(u2_p99) model
    tail_rows = [r for r in rows if r["u2_p99_at_stepOff"] and r["inject_repeat_nsteps"] and r["driveAmp_scale"]]
    tail_fit = None
    tail_residuals = []
    tail_max_abs_log_err = None
    tail_rmse = None
    if len(tail_rows) >= 2:
        y = [r["u2_p99_at_stepOff"] for r in tail_rows]
        if args.tail_model == "logquad2d":
            X = [[math.log(r["repeat_pre"] if args.repeat_pre and r.get("repeat_pre") is not None else r["inject_repeat_nsteps"]),
                  math.log(r["driveAmp_scale"]),
                  math.log(r["repeat_pre"] if args.repeat_pre and r.get("repeat_pre") is not None else r["inject_repeat_nsteps"]) ** 2,
                  math.log(r["driveAmp_scale"]) ** 2,
                  math.log(r["repeat_pre"] if args.repeat_pre and r.get("repeat_pre") is not None else r["inject_repeat_nsteps"]) * math.log(r["driveAmp_scale"])] for r in tail_rows]
            tail_fit = fit_log_linear(y, X, ["log_repeat", "log_driveAmp", "log_repeat2", "log_driveAmp2", "log_repeat_log_drive"])
        elif args.tail_model == "logquad":
            X = [[math.log(r["repeat_pre"] if args.repeat_pre and r.get("repeat_pre") is not None else r["inject_repeat_nsteps"]),
                  math.log(r["repeat_pre"] if args.repeat_pre and r.get("repeat_pre") is not None else r["inject_repeat_nsteps"]) ** 2,
                  math.log(r["driveAmp_scale"])] for r in tail_rows]
            tail_fit = fit_log_linear(y, X, ["log_repeat", "log_repeat2", "log_driveAmp"])
        elif args.tail_model == "piecewise_Lpre":
            knot = args.tail_knot
            L0 = math.log(knot)
            X = []
            for r in tail_rows:
                repeat_val = r["repeat_pre"] if args.repeat_pre and r.get("repeat_pre") is not None else r["inject_repeat_nsteps"]
                L = math.log(repeat_val)
                X.append([L, math.log(r["driveAmp_scale"]), max(0.0, L - L0)])
            tail_fit = fit_log_linear(y, X, ["log_repeat", "log_driveAmp", "log_repeat_kink"])
        else:
            X = [[math.log(r["repeat_pre"] if args.repeat_pre and r.get("repeat_pre") is not None else r["inject_repeat_nsteps"]), math.log(r["driveAmp_scale"])] for r in tail_rows]
            tail_fit = fit_log_linear(y, X, ["log_repeat", "log_driveAmp"])

        if tail_fit:
            coeffs = tail_fit["coeffs"]
            abs_errs = []
            for r in tail_rows:
                L = math.log(r["repeat_pre"] if args.repeat_pre and r.get("repeat_pre") is not None else r["inject_repeat_nsteps"])
                ld = math.log(r["driveAmp_scale"])
                if args.tail_model == "logquad2d":
                    pred_log = (coeffs["intercept"] +
                                coeffs["log_repeat"] * L +
                                coeffs["log_driveAmp"] * ld +
                                coeffs["log_repeat2"] * (L ** 2) +
                                coeffs["log_driveAmp2"] * (ld ** 2) +
                                coeffs["log_repeat_log_drive"] * (L * ld))
                elif args.tail_model == "logquad":
                    pred_log = coeffs["intercept"] + coeffs["log_repeat"] * L + coeffs["log_repeat2"] * (L ** 2) + coeffs["log_driveAmp"] * ld
                elif args.tail_model == "piecewise_Lpre":
                    L0 = math.log(args.tail_knot)
                    pred_log = (coeffs["intercept"] +
                                coeffs["log_repeat"] * L +
                                coeffs["log_driveAmp"] * ld +
                                coeffs["log_repeat_kink"] * max(0.0, L - L0))
                else:
                    pred_log = coeffs["intercept"] + coeffs["log_repeat"] * L + coeffs["log_driveAmp"] * ld
                true_log = math.log(r["u2_p99_at_stepOff"])
                err = pred_log - true_log
                abs_errs.append(abs(err))
                tail_residuals.append({"case": r["case"], "log_err": err})
            if abs_errs:
                tail_max_abs_log_err = max(abs_errs)
                tail_rmse = math.sqrt(sum(e * e for e in abs_errs) / len(abs_errs))

    # Circuit fit: log(e_load) ~ a + b*log(driveAmp) + d*log(repeat)
    circuit_rows = [r for r in rows if r["e_load"] and r["driveAmp_scale"] and r["inject_repeat_nsteps"]]
    circuit_fit = None
    if len(circuit_rows) >= 2:
        y = [r["e_load"] for r in circuit_rows]
        X = [[math.log(r["driveAmp_scale"]), math.log(r["inject_repeat_nsteps"])] for r in circuit_rows]
        circuit_fit = fit_log_linear(y, X, ["log_driveAmp", "log_repeat"])

    # Formation phase fit: log(rho_delta_rel_phase) model
    rho_rows = [r for r in rows if r["rho_delta_rel_phase"] and r["inject_repeat_nsteps"] and r["driveAmp_scale"]]
    rho_fit = None
    rho_residuals = []
    rho_max_abs_log_err = None
    rho_rmse = None
    if len(rho_rows) >= 2:
        y = [r["rho_delta_rel_phase"] for r in rho_rows]
        if args.rho_model == "logquad2d":
            X = [[math.log(r["repeat_pre"] if args.repeat_pre and r.get("repeat_pre") is not None else r["inject_repeat_nsteps"]),
                  math.log(r["driveAmp_scale"]),
                  math.log(r["repeat_pre"] if args.repeat_pre and r.get("repeat_pre") is not None else r["inject_repeat_nsteps"]) ** 2,
                  math.log(r["driveAmp_scale"]) ** 2,
                  math.log(r["repeat_pre"] if args.repeat_pre and r.get("repeat_pre") is not None else r["inject_repeat_nsteps"]) * math.log(r["driveAmp_scale"])] for r in rho_rows]
            rho_fit = fit_log_linear(y, X, ["log_repeat", "log_driveAmp", "log_repeat2", "log_driveAmp2", "log_repeat_log_drive"])
        elif args.rho_model == "logquad":
            X = [[math.log(r["repeat_pre"] if args.repeat_pre and r.get("repeat_pre") is not None else r["inject_repeat_nsteps"]),
                  math.log(r["repeat_pre"] if args.repeat_pre and r.get("repeat_pre") is not None else r["inject_repeat_nsteps"]) ** 2,
                  math.log(r["driveAmp_scale"])] for r in rho_rows]
            rho_fit = fit_log_linear(y, X, ["log_repeat", "log_repeat2", "log_driveAmp"])
        else:
            X = [[math.log(r["repeat_pre"] if args.repeat_pre and r.get("repeat_pre") is not None else r["inject_repeat_nsteps"]), math.log(r["driveAmp_scale"])] for r in rho_rows]
            rho_fit = fit_log_linear(y, X, ["log_repeat", "log_driveAmp"])

        # Compute residuals in log space for rho fit
        if rho_fit:
            coeffs = rho_fit["coeffs"]
            abs_errs = []
            for r in rho_rows:
                L = math.log(r["repeat_pre"] if args.repeat_pre and r.get("repeat_pre") is not None else r["inject_repeat_nsteps"])
                ld = math.log(r["driveAmp_scale"])
                if args.rho_model == "logquad2d":
                    pred_log = (coeffs["intercept"] +
                                coeffs["log_repeat"] * L +
                                coeffs["log_driveAmp"] * ld +
                                coeffs["log_repeat2"] * (L ** 2) +
                                coeffs["log_driveAmp2"] * (ld ** 2) +
                                coeffs["log_repeat_log_drive"] * (L * ld))
                elif args.rho_model == "logquad":
                    pred_log = coeffs["intercept"] + coeffs["log_repeat"] * L + coeffs["log_repeat2"] * (L ** 2) + coeffs["log_driveAmp"] * ld
                else:
                    pred_log = coeffs["intercept"] + coeffs["log_repeat"] * L + coeffs["log_driveAmp"] * ld
                true_log = math.log(r["rho_delta_rel_phase"])
                err = pred_log - true_log
                abs_errs.append(abs(err))
                rho_residuals.append({"case": r["case"], "log_err": err})
            if abs_errs:
                rho_max_abs_log_err = max(abs_errs)
                rho_rmse = math.sqrt(sum(e * e for e in abs_errs) / len(abs_errs))

    rho_strong = None
    rho_strong_fit = None
    if args.rho_strong_clf == "logistic" and len(rho_rows) >= 2:
        # Logistic classifier for strong compression: rho >= threshold
        # Features: basic=[1, Lpre, Ld, Lpre^2], quad adds Ld^2 and Lpre*Ld.
        import numpy as _np
        X = []
        yb = []
        for r in rho_rows:
            Lpre = math.log(r["repeat_pre"] if args.repeat_pre and r.get("repeat_pre") is not None else r["inject_repeat_nsteps"])
            Ld = math.log(r["driveAmp_scale"])
            row = [1.0, Lpre, Ld, Lpre ** 2]
            if args.rho_strong_features == "quad":
                row.append(Ld ** 2)
                row.append(Lpre * Ld)
            X.append(row)
            yb.append(1.0 if r["rho_delta_rel_phase"] >= args.rho_strong_threshold else 0.0)
        X = _np.array(X, dtype=float)
        yb = _np.array(yb, dtype=float)
        # Newton-Raphson logistic regression
        w = _np.zeros(X.shape[1])
        for _ in range(50):
            z = X @ w
            p = 1.0 / (1.0 + _np.exp(-z))
            W = _np.diag(p * (1 - p))
            grad = X.T @ (p - yb)
            H = X.T @ W @ X
            try:
                step = _np.linalg.solve(H, grad)
            except _np.linalg.LinAlgError:
                break
            w = w - step
        # simple accuracy + AUC
        p = 1.0 / (1.0 + _np.exp(-(X @ w)))
        pred = (p >= 0.5).astype(float)
        acc = float((pred == yb).mean())
        # AUC via rank statistic
        try:
            order = _np.argsort(p)
            ranks = _np.empty_like(order, dtype=float)
            ranks[order] = _np.arange(1, len(p) + 1)
            n_pos = int((yb == 1).sum())
            n_neg = int((yb == 0).sum())
            auc = None
            if n_pos > 0 and n_neg > 0:
                auc = float((ranks[yb == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
        except Exception:
            auc = None
            n_pos = int((yb == 1).sum())
            n_neg = int((yb == 0).sum())
        # confusion matrix
        tp = int(((pred == 1) & (yb == 1)).sum())
        fp = int(((pred == 1) & (yb == 0)).sum())
        tn = int(((pred == 0) & (yb == 0)).sum())
        fn = int(((pred == 0) & (yb == 1)).sum())
        features = ["1", "log_repeat_pre", "log_driveAmp", "log_repeat_pre2"]
        coef = {"intercept": float(w[0]), "log_repeat": float(w[1]), "log_driveAmp": float(w[2]), "log_repeat2": float(w[3])}
        if args.rho_strong_features == "quad":
            features += ["log_driveAmp2", "log_repeat_pre_log_drive"]
            coef["log_driveAmp2"] = float(w[4])
            coef["log_repeat_log_drive"] = float(w[5])
        # audit inputs
        model_inputs = []
        for r in rho_rows:
            metrics_path = Path("outputs") / r["case"] / "analysis" / "metrics_formation.json"
            sha1 = None
            if metrics_path.exists():
                h = hashlib.sha1()
                h.update(metrics_path.read_bytes())
                sha1 = h.hexdigest()
            model_inputs.append({"case": r["case"], "metrics_formation_sha1": sha1})
        rho_strong = {
            "kind": "logistic",
            "features": features,
            "coef": coef,
            "accuracy": acc,
            "auc": auc,
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "threshold": args.rho_strong_threshold,
            "n_samples": int(len(yb)),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "sample_small": bool(len(yb) < 20),
            "model_inputs_sha1": model_inputs,
        }

        # Strong-only regression (reuse rho-model) on strong samples
        strong_rows = [r for r in rho_rows if r["rho_delta_rel_phase"] >= args.rho_strong_threshold]
        if len(strong_rows) >= 2:
            y_strong = [r["rho_delta_rel_phase"] for r in strong_rows]
            if args.rho_model == "logquad2d":
                Xs = [[math.log(r["repeat_pre"] if args.repeat_pre and r.get("repeat_pre") is not None else r["inject_repeat_nsteps"]),
                       math.log(r["driveAmp_scale"]),
                       math.log(r["repeat_pre"] if args.repeat_pre and r.get("repeat_pre") is not None else r["inject_repeat_nsteps"]) ** 2,
                       math.log(r["driveAmp_scale"]) ** 2,
                       math.log(r["repeat_pre"] if args.repeat_pre and r.get("repeat_pre") is not None else r["inject_repeat_nsteps"]) * math.log(r["driveAmp_scale"])] for r in strong_rows]
                rho_strong_fit = fit_log_linear(y_strong, Xs, ["log_repeat", "log_driveAmp", "log_repeat2", "log_driveAmp2", "log_repeat_log_drive"])
            elif args.rho_model == "logquad":
                Xs = [[math.log(r["repeat_pre"] if args.repeat_pre and r.get("repeat_pre") is not None else r["inject_repeat_nsteps"]),
                       math.log(r["repeat_pre"] if args.repeat_pre and r.get("repeat_pre") is not None else r["inject_repeat_nsteps"]) ** 2,
                       math.log(r["driveAmp_scale"])] for r in strong_rows]
                rho_strong_fit = fit_log_linear(y_strong, Xs, ["log_repeat", "log_repeat2", "log_driveAmp"])
            else:
                Xs = [[math.log(r["repeat_pre"] if args.repeat_pre and r.get("repeat_pre") is not None else r["inject_repeat_nsteps"]), math.log(r["driveAmp_scale"])] for r in strong_rows]
                rho_strong_fit = fit_log_linear(y_strong, Xs, ["log_repeat", "log_driveAmp"])

    out = {
        "cases": rows,
        "model_tail": {
            "formula": (
                "log(u2_p99_at_stepOff) = a + b*log(repeat) + c*log(driveAmp)"
                if args.tail_model == "loglin"
                else "log(u2_p99_at_stepOff) = a + b*log(repeat) + d*log(repeat)^2 + c*log(driveAmp)"
                if args.tail_model == "logquad"
                else "log(u2_p99_at_stepOff) = a + b*log(repeat) + c*log(driveAmp) + d*log(repeat)^2 + e*log(driveAmp)^2 + f*log(repeat)*log(driveAmp)"
                if args.tail_model == "logquad2d"
                else "log(u2_p99_at_stepOff) = a + b*log(repeat) + c*log(driveAmp) + k*max(0, log(repeat)-log(repeat_knot))"
            ),
            "model_kind": (
                "loglin"
                if args.tail_model == "loglin"
                else "logquad"
                if args.tail_model == "logquad"
                else "logquad2d"
                if args.tail_model == "logquad2d"
                else "piecewise_Lpre"
            ),
            "fit": tail_fit,
            "repeat_feature": "repeat_pre" if args.repeat_pre else "repeat",
            "repeat_knot": float(args.tail_knot) if args.tail_model == "piecewise_Lpre" else None,
            "log_repeat_knot": float(math.log(args.tail_knot)) if args.tail_model == "piecewise_Lpre" else None,
            "max_abs_log_err": None if tail_max_abs_log_err is None else float(tail_max_abs_log_err),
            "rmse_log": None if tail_rmse is None else float(tail_rmse),
            "residuals": tail_residuals,
        },
        "model_circuit": {
            "formula": "log(e_load) = a + b*log(driveAmp) + d*log(repeat)",
            "fit": circuit_fit,
        },
        "model_formation_phase": {
            "formula": (
                "log(rho_delta_rel_phase) = a + b*log(repeat) + c*log(driveAmp)"
                if args.rho_model == "loglin"
                else "log(rho_delta_rel_phase) = a + b*log(repeat) + d*log(repeat)^2 + c*log(driveAmp)"
                if args.rho_model == "logquad"
                else "log(rho_delta_rel_phase) = a + b*log(repeat) + c*log(driveAmp) + d*log(repeat)^2 + e*log(driveAmp)^2 + f*log(repeat)*log(driveAmp)"
            ),
            "model_kind": "loglin" if args.rho_model == "loglin" else "logquad" if args.rho_model == "logquad" else "logquad2d",
            "fit": rho_fit,
            "repeat_feature": "repeat_pre" if args.repeat_pre else "repeat",
            "max_abs_log_err": None if rho_max_abs_log_err is None else float(rho_max_abs_log_err),
            "rmse_log": None if rho_rmse is None else float(rho_rmse),
            "residuals": rho_residuals,
        },
        "rho_strong_classifier": rho_strong,
        "rho_strong_regression": rho_strong_fit,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        import csv
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)


if __name__ == "__main__":
    main()
