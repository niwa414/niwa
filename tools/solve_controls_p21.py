#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text())


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    h.update(path.read_bytes())
    return h.hexdigest()


def clamp(val, vmin, vmax):
    return max(vmin, min(vmax, val))


def safe_log(x):
    if x is None:
        return None
    if x <= 0:
        raise ValueError(f"log undefined for non-positive value: {x}")
    return math.log(x)


def predict_from_model(model, drive_amp, repeat):
    preds = {}
    # circuit: log(e_load) = a + b*log(driveAmp) [+ d*log(repeat)]
    circuit = model.get("model_circuit", {}).get("fit")
    if circuit:
        a = circuit["coeffs"]["intercept"]
        b = circuit["coeffs"]["log_driveAmp"]
        d = circuit["coeffs"].get("log_repeat", 0.0)
        preds["e_load_pred"] = math.exp(a + b * math.log(drive_amp) + d * math.log(repeat))
    # tail: log(u2) model
    tail_meta = model.get("model_tail", {})
    tail = tail_meta.get("fit")
    if tail:
        a = tail["coeffs"]["intercept"]
        b = tail["coeffs"]["log_repeat"]
        c = tail["coeffs"]["log_driveAmp"]
        d = tail["coeffs"].get("log_repeat2", 0.0)
        L = math.log(repeat)
        if tail_meta.get("model_kind") == "piecewise_Lpre":
            L0 = tail_meta.get("log_repeat_knot", 0.0)
            kink = tail["coeffs"].get("log_repeat_kink", 0.0)
            preds["u2_pred"] = math.exp(a + b * L + c * math.log(drive_amp) + kink * max(0.0, L - L0))
        else:
            preds["u2_pred"] = math.exp(a + b * L + d * (L ** 2) + c * math.log(drive_amp))
    # phase compression: log(rho) = a + b*log(repeat) [+ d*log(repeat)^2] [+ c*log(driveAmp)]
    rho = model.get("model_formation_phase", {}).get("fit")
    if rho:
        a = rho["coeffs"]["intercept"]
        b = rho["coeffs"]["log_repeat"]
        c = rho["coeffs"].get("log_driveAmp", 0.0)
        d = rho["coeffs"].get("log_repeat2", 0.0)
        L = math.log(repeat)
        preds["rho_pred"] = math.exp(a + b * L + d * (L ** 2) + c * math.log(drive_amp))
    return preds


def grid_search_repeat(model, drive, targets, repeat_min, repeat_max, w_u2, w_rho, rho_mode, rho_threshold, penalty):
    tail_meta = model.get("model_tail", {})
    tail = tail_meta.get("fit")
    rho = model.get("model_formation_phase", {}).get("fit")
    if tail is None and rho is None:
        return repeat_min
    best_repeat = repeat_min
    best_J = None
    log_drive = math.log(drive)
    for repeat in range(int(repeat_min), int(repeat_max) + 1):
        L = math.log(repeat)
        J = 0.0
        if targets.get("u2_target") is not None and tail is not None:
            aU = tail["coeffs"]["intercept"]
            bU = tail["coeffs"]["log_repeat"]
            cU = tail["coeffs"]["log_driveAmp"]
            dU = tail["coeffs"].get("log_repeat2", 0.0)
            if tail_meta.get("model_kind") == "piecewise_Lpre":
                L0 = tail_meta.get("log_repeat_knot", 0.0)
                kink = tail["coeffs"].get("log_repeat_kink", 0.0)
                log_u2_pred = aU + bU * L + cU * log_drive + kink * max(0.0, L - L0)
            else:
                log_u2_pred = aU + bU * L + dU * (L ** 2) + cU * log_drive
            log_u2_t = safe_log(targets["u2_target"])
            J += w_u2 * (log_u2_pred - log_u2_t) ** 2
        if targets.get("rho_target") is not None and rho is not None:
            aR = rho["coeffs"]["intercept"]
            bR = rho["coeffs"]["log_repeat"]
            dR = rho["coeffs"].get("log_repeat2", 0.0)
            cR = rho["coeffs"].get("log_driveAmp", 0.0)
            log_rho_pred = aR + bR * L + dR * (L ** 2) + cR * log_drive
            log_rho_t = safe_log(targets["rho_target"])
            if rho_mode == "threshold":
                if math.exp(log_rho_pred) < targets["rho_target"]:
                    J += penalty * (log_rho_t - log_rho_pred) ** 2
            else:
                J += w_rho * (log_rho_pred - log_rho_t) ** 2
        if best_J is None or J < best_J:
            best_J = J
            best_repeat = repeat
    return best_repeat


def inverse_controls(model, targets, drive_default, repeat_default, drive_min, drive_max, repeat_min, repeat_max, w_u2, w_rho, rho_mode):
    # Solve for driveAmp from e_load if provided
    drive = drive_default
    if targets.get("e_load_target") is not None:
        circuit = model.get("model_circuit", {}).get("fit")
        a = circuit["coeffs"]["intercept"]
        b = circuit["coeffs"]["log_driveAmp"]
        d = circuit["coeffs"].get("log_repeat", 0.0)
        drive = math.exp((safe_log(targets["e_load_target"]) - a - d * math.log(repeat_default)) / b)
        drive = clamp(drive, drive_min, drive_max)

    # Solve for repeat from u2/rho
    repeat = repeat_default
    tail = model.get("model_tail", {}).get("fit")
    rho = model.get("model_formation_phase", {}).get("fit")
    have_u2 = targets.get("u2_target") is not None and tail is not None
    have_rho = targets.get("rho_target") is not None and rho is not None

    if rho_mode == "threshold" and (have_u2 or have_rho):
        repeat_u2 = None
        repeat_rho = None
        if have_u2:
            a = tail["coeffs"]["intercept"]
            b = tail["coeffs"]["log_repeat"]
            c = tail["coeffs"]["log_driveAmp"]
            repeat_u2 = math.exp((safe_log(targets["u2_target"]) - a - c * math.log(drive)) / b)
        if have_rho:
            a = rho["coeffs"]["intercept"]
            b = rho["coeffs"]["log_repeat"]
            c = rho["coeffs"].get("log_driveAmp", 0.0)
            repeat_rho = math.exp((safe_log(targets["rho_target"]) - a - c * math.log(drive)) / b)
        if repeat_u2 is None and repeat_rho is not None:
            repeat = repeat_rho
        elif repeat_rho is None and repeat_u2 is not None:
            repeat = repeat_u2
        elif repeat_u2 is not None and repeat_rho is not None:
            repeat = max(repeat_u2, repeat_rho)
    elif have_u2 and not have_rho:
        a = tail["coeffs"]["intercept"]
        b = tail["coeffs"]["log_repeat"]
        c = tail["coeffs"]["log_driveAmp"]
        repeat = math.exp((safe_log(targets["u2_target"]) - a - c * math.log(drive)) / b)
    elif have_rho and not have_u2:
        a = rho["coeffs"]["intercept"]
        b = rho["coeffs"]["log_repeat"]
        c = rho["coeffs"].get("log_driveAmp", 0.0)
        repeat = math.exp((safe_log(targets["rho_target"]) - a - c * math.log(drive)) / b)
    elif have_u2 and have_rho:
        aU = tail["coeffs"]["intercept"]
        bU = tail["coeffs"]["log_repeat"]
        cU = tail["coeffs"]["log_driveAmp"]
        aR = rho["coeffs"]["intercept"]
        bR = rho["coeffs"]["log_repeat"]
        cR = rho["coeffs"].get("log_driveAmp", 0.0)
        log_drive = math.log(drive)
        num = (w_u2 * bU * (safe_log(targets["u2_target"]) - aU - cU * log_drive) +
               w_rho * bR * (safe_log(targets["rho_target"]) - aR - cR * log_drive))
        den = w_u2 * (bU ** 2) + w_rho * (bR ** 2)
        repeat = math.exp(num / den)

    repeat = clamp(repeat, repeat_min, repeat_max)
    # Output integer repeat
    repeat_int = int(round(repeat))
    repeat_int = int(clamp(repeat_int, repeat_min, repeat_max))

    # If circuit model includes repeat term and e_load_target was provided, refine drive with solved repeat
    if targets.get("e_load_target") is not None:
        circuit = model.get("model_circuit", {}).get("fit")
        d = circuit["coeffs"].get("log_repeat", 0.0)
        if d != 0.0:
            a = circuit["coeffs"]["intercept"]
            b = circuit["coeffs"]["log_driveAmp"]
            drive = math.exp((safe_log(targets["e_load_target"]) - a - d * math.log(repeat_int)) / b)
            drive = clamp(drive, drive_min, drive_max)
    return drive, repeat_int


def compute_errors(targets, preds):
    errors = {}
    for key, tgt in targets.items():
        if tgt is None:
            continue
        pred_key = key.replace("_target", "_pred")
        pred = preds.get(pred_key)
        if pred is None:
            continue
        errors[pred_key + "_rel_err"] = (pred - tgt) / tgt
        errors[pred_key + "_log_err"] = math.log(pred) - math.log(tgt)
    return errors


def self_check_table(model, cases, drive_default, repeat_default, drive_min, drive_max, repeat_min, repeat_max, w_u2, w_rho, rho_threshold, rho_mode, repeat_search, penalty):
    rows = []
    max_rel = {"e_load_pred": 0.0, "u2_pred": 0.0, "rho_pred": 0.0}
    max_abs_log = {"u2_pred": 0.0, "rho_pred": 0.0}
    max_rel_rho_thresh = 0.0
    for c in cases:
        targets = {
            "e_load_target": c.get("e_load"),
            "u2_target": c.get("u2_p99_at_stepOff"),
            "rho_target": c.get("rho_delta_rel_phase"),
        }
        if repeat_search == "grid":
            drive, _ = inverse_controls(
                model, targets, drive_default, repeat_default, drive_min, drive_max, repeat_min, repeat_max, w_u2, w_rho, rho_mode
            )
            repeat = grid_search_repeat(
                model, drive, targets, int(repeat_min), int(repeat_max),
                w_u2, w_rho, rho_mode, rho_threshold, penalty
            )
        else:
            drive, repeat = inverse_controls(
                model, targets, drive_default, repeat_default, drive_min, drive_max, repeat_min, repeat_max, w_u2, w_rho, rho_mode
            )
        preds = predict_from_model(model, drive, repeat)
        errors = compute_errors(targets, preds)
        for k, v in errors.items():
            if k.endswith("_rel_err"):
                key = k.replace("_rel_err", "")
                max_rel[key] = max(max_rel[key], abs(v))
            if k.endswith("_log_err"):
                key = k.replace("_log_err", "")
                if key in max_abs_log:
                    max_abs_log[key] = max(max_abs_log[key], abs(v))
        rho_target = targets.get("rho_target")
        if rho_target is not None and rho_target >= rho_threshold:
            rel_key = "rho_pred_rel_err"
            if rel_key in errors:
                max_rel_rho_thresh = max(max_rel_rho_thresh, abs(errors[rel_key]))
        rows.append({
            "case": c.get("case"),
            "driveAmp_actual": c.get("driveAmp_scale"),
            "repeat_actual": c.get("inject_repeat_nsteps"),
            "driveAmp_reco": drive,
            "repeat_reco": repeat,
            **preds,
            **errors,
        })
    return rows, max_rel, max_abs_log, max_rel_rho_thresh


def main():
    parser = argparse.ArgumentParser(description="Inverse control solver for P21.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--eload", type=float)
    parser.add_argument("--u2", type=float)
    parser.add_argument("--rho", type=float)
    parser.add_argument("--drive-default", type=float, default=1.0)
    parser.add_argument("--repeat-default", type=float, default=256.0)
    parser.add_argument("--drive-min", type=float, default=0.1)
    parser.add_argument("--drive-max", type=float, default=3.0)
    parser.add_argument("--repeat-min", type=float, default=8.0)
    parser.add_argument("--repeat-max", type=float, default=512.0)
    parser.add_argument("--w_u2", type=float, default=1.0)
    parser.add_argument("--w_rho", type=float, default=1.0)
    parser.add_argument("--rho-mode", choices=["exact", "threshold"], default="exact")
    parser.add_argument("--rho-threshold", type=float, default=0.05)
    parser.add_argument("--repeat-search", choices=["grid", "closed"], default="grid")
    parser.add_argument("--penalty", type=float, default=100.0)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-out", default="outputs/p25-solver-tune/analysis/p25_sweep.csv")
    parser.add_argument("--best-out", default="outputs/p25-solver-tune/analysis/metrics_p25_best.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model)
    model = load_json(model_path)

    targets = {
        "e_load_target": args.eload,
        "u2_target": args.u2,
        "rho_target": args.rho,
    }

    if args.sweep:
        # Hyperparameter sweep to reduce u2 error while keeping rho error bounded.
        w_u2_grid = [0.5, 1.0, 2.0, 4.0]
        w_rho_grid = [0.5, 1.0, 2.0, 4.0]
        rho_mode_grid = ["exact", "threshold"]
        penalty_grid = [25, 50, 100, 200]

        cases = model.get("cases", [])
        rows = []
        best = None
        for w_u2 in w_u2_grid:
            for w_rho in w_rho_grid:
                for rho_mode in rho_mode_grid:
                    penalties = penalty_grid if rho_mode == "threshold" else [args.penalty]
                    for penalty in penalties:
                        _, max_rel, _, max_rel_rho_thresh = self_check_table(
                            model, cases, args.drive_default, args.repeat_default,
                            args.drive_min, args.drive_max, args.repeat_min, args.repeat_max,
                            w_u2, w_rho, args.rho_threshold, rho_mode, "grid", penalty
                        )
                        max_rel_u2 = max_rel.get("u2_pred")
                        score = None if max_rel_u2 is None else max_rel_u2 + max_rel_rho_thresh
                        passes = max_rel_rho_thresh is not None and max_rel_rho_thresh <= 0.20
                        row = {
                            "w_u2": w_u2,
                            "w_rho": w_rho,
                            "rho_mode": rho_mode,
                            "penalty": penalty,
                            "max_rel_err_u2": max_rel_u2,
                            "max_rel_err_rho_ge_0p05": max_rel_rho_thresh,
                            "score": score,
                            "passes_rho_constraint": passes,
                        }
                        rows.append(row)
                        if score is None:
                            continue
                        if best is None:
                            best = row
                        else:
                            # Prefer candidates that pass rho constraint; then lowest score.
                            if row["passes_rho_constraint"] and not best["passes_rho_constraint"]:
                                best = row
                            elif row["passes_rho_constraint"] == best["passes_rho_constraint"]:
                                if row["score"] < best["score"]:
                                    best = row

        sweep_out = Path(args.sweep_out)
        sweep_out.parent.mkdir(parents=True, exist_ok=True)
        import csv as _csv
        with sweep_out.open("w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=sorted(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)

        best_out = Path(args.best_out)
        best_out.parent.mkdir(parents=True, exist_ok=True)
        best_out.write_text(json.dumps({
            "best": best,
            "rho_threshold": args.rho_threshold,
        }, indent=2))
        return
    if args.repeat_search == "grid":
        # Solve drive first (closed form) then grid search repeat
        drive, _ = inverse_controls(
            model, targets, args.drive_default, args.repeat_default,
            args.drive_min, args.drive_max, args.repeat_min, args.repeat_max,
            args.w_u2, args.w_rho, args.rho_mode,
        )
        repeat = grid_search_repeat(
            model, drive, targets, int(args.repeat_min), int(args.repeat_max),
            args.w_u2, args.w_rho, args.rho_mode, args.rho_threshold, args.penalty
        )
    else:
        drive, repeat = inverse_controls(
            model, targets, args.drive_default, args.repeat_default,
            args.drive_min, args.drive_max, args.repeat_min, args.repeat_max,
            args.w_u2, args.w_rho, args.rho_mode,
        )
    preds = predict_from_model(model, drive, repeat)
    errors = compute_errors(targets, preds)

    # Self-check on DOE cases if requested
    self_check_rows = []
    max_rel = {}
    if args.self_check:
        cases = model.get("cases", [])
        self_check_rows, max_rel, max_abs_log, max_rel_rho_thresh = self_check_table(
            model, cases, args.drive_default, args.repeat_default,
            args.drive_min, args.drive_max, args.repeat_min, args.repeat_max,
            args.w_u2, args.w_rho, args.rho_threshold, args.rho_mode, args.repeat_search, args.penalty
        )
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(self_check_rows[0].keys()) if self_check_rows else [])
            if self_check_rows:
                writer.writeheader()
                writer.writerows(self_check_rows)

    out = {
        "inputs": {
            **targets,
            "drive_default": args.drive_default,
            "repeat_default": args.repeat_default,
            "drive_min": args.drive_min,
            "drive_max": args.drive_max,
            "repeat_min": args.repeat_min,
            "repeat_max": args.repeat_max,
            "w_u2": args.w_u2,
            "w_rho": args.w_rho,
            "rho_mode": args.rho_mode,
            "rho_threshold": args.rho_threshold,
            "repeat_search": args.repeat_search,
            "penalty": args.penalty,
        },
        "recommend": {
            "driveAmp": drive,
            "inject_repeat_nsteps": repeat,
        },
        "predict": preds,
        "errors": errors,
        "model_audit": {
            "path": str(model_path),
            "sha1": sha1_file(model_path),
            "coeffs": {
                "circuit": model.get("model_circuit", {}).get("fit", {}).get("coeffs"),
                "tail": model.get("model_tail", {}).get("fit", {}).get("coeffs"),
                "phase_compression": model.get("model_formation_phase", {}).get("fit", {}).get("coeffs"),
            },
        },
    }

    if args.self_check:
        out["self_check"] = {
            "rows": self_check_rows,
            "max_rel_err": max_rel,
            "max_abs_log_err": max_abs_log,
            "max_rel_err_rho_ge_threshold": max_rel_rho_thresh,
            "rho_threshold": args.rho_threshold,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
