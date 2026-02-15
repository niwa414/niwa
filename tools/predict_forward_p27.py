#!/usr/bin/env python3
import argparse
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


def main():
    parser = argparse.ArgumentParser(description="Forward prediction from control-law model (P27).")
    parser.add_argument("--model", required=True)
    parser.add_argument("--driveAmp", type=float, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--off-step", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model_path = Path(args.model)
    model = load_json(model_path)
    drive = args.driveAmp
    repeat = args.repeat
    off_step = args.off_step
    L = math.log(repeat)
    log_drive = math.log(drive)

    # Circuit
    e_load_pred = None
    circuit = model.get("model_circuit", {}).get("fit")
    if circuit:
        a = circuit["coeffs"]["intercept"]
        b = circuit["coeffs"]["log_driveAmp"]
        d = circuit["coeffs"].get("log_repeat", 0.0)
        e_load_pred = math.exp(a + b * log_drive + d * L)

    # Tail
    u2_pred = None
    tail_meta = model.get("model_tail", {})
    tail = tail_meta.get("fit")
    if tail:
        repeat_feature = tail_meta.get("repeat_feature", "repeat")
        if repeat_feature == "repeat_pre" and off_step is not None:
            repeat_used = min(repeat, off_step)
        else:
            repeat_used = repeat
        a = tail["coeffs"]["intercept"]
        b = tail["coeffs"]["log_repeat"]
        c = tail["coeffs"]["log_driveAmp"]
        d = tail["coeffs"].get("log_repeat2", 0.0)
        e = tail["coeffs"].get("log_driveAmp2", 0.0)
        f = tail["coeffs"].get("log_repeat_log_drive", 0.0)
        L = math.log(repeat_used)
        if tail_meta.get("model_kind") == "piecewise_Lpre":
            L0 = tail_meta.get("log_repeat_knot")
            kink = tail["coeffs"].get("log_repeat_kink", 0.0)
            u2_pred = math.exp(a + b * L + c * log_drive + kink * max(0.0, L - L0))
        else:
            u2_pred = math.exp(a + b * L + d * (L ** 2) + c * log_drive + e * (log_drive ** 2) + f * (L * log_drive))

    # Phase compression
    rho_pred = None
    rho = model.get("model_formation_phase", {}).get("fit")
    if rho:
        repeat_feature = model.get("model_formation_phase", {}).get("repeat_feature", "repeat")
        if repeat_feature == "repeat_pre" and off_step is not None:
            repeat_used = min(repeat, off_step)
        else:
            repeat_used = repeat
        a = rho["coeffs"]["intercept"]
        b = rho["coeffs"]["log_repeat"]
        c = rho["coeffs"].get("log_driveAmp", 0.0)
        d = rho["coeffs"].get("log_repeat2", 0.0)
        e = rho["coeffs"].get("log_driveAmp2", 0.0)
        f = rho["coeffs"].get("log_repeat_log_drive", 0.0)
        L = math.log(repeat_used)
        rho_pred = math.exp(a + b * L + d * (L ** 2) + c * log_drive + e * (log_drive ** 2) + f * (L * log_drive))

    # Override rho with strong classifier if present
    rho_mode = "regression"
    rho_strong_prob = None
    rho_pred_ge = None
    clf = model.get("rho_strong_classifier")
    strong_fit = model.get("rho_strong_regression")
    if clf and off_step is not None:
        repeat_used = min(repeat, off_step) if model.get("model_formation_phase", {}).get("repeat_feature") == "repeat_pre" else repeat
        L = math.log(repeat_used)
        ld = log_drive
        features = clf.get("features", [])
        coef = clf.get("coef", {})
        z = coef.get("intercept", 0.0)
        for name in features:
            if name == "1":
                continue
            if name == "log_repeat_pre" or name == "log_repeat":
                z += coef.get("log_repeat", 0.0) * L
            elif name == "log_driveAmp":
                z += coef.get("log_driveAmp", 0.0) * ld
            elif name == "log_repeat_pre2" or name == "log_repeat2":
                z += coef.get("log_repeat2", 0.0) * (L ** 2)
            elif name == "log_driveAmp2":
                z += coef.get("log_driveAmp2", 0.0) * (ld ** 2)
            elif name == "log_repeat_pre_log_drive" or name == "log_repeat_log_drive":
                z += coef.get("log_repeat_log_drive", 0.0) * (L * ld)
        rho_strong_prob = 1.0 / (1.0 + math.exp(-z))
        threshold = clf.get("threshold", 0.5)
        rho_pred_ge = rho_strong_prob >= threshold
        if strong_fit and rho_pred_ge:
            coef_r = strong_fit["coeffs"]
            if "log_repeat2" in coef_r and "log_driveAmp2" in coef_r and "log_repeat_log_drive" in coef_r:
                rho_pred = math.exp(
                    coef_r["intercept"] +
                    coef_r["log_repeat"] * L +
                    coef_r["log_driveAmp"] * ld +
                    coef_r["log_repeat2"] * (L ** 2) +
                    coef_r["log_driveAmp2"] * (ld ** 2) +
                    coef_r["log_repeat_log_drive"] * (L * ld)
                )
            elif "log_repeat2" in coef_r:
                rho_pred = math.exp(
                    coef_r["intercept"] +
                    coef_r["log_repeat"] * L +
                    coef_r["log_repeat2"] * (L ** 2) +
                    coef_r.get("log_driveAmp", 0.0) * ld
                )
            else:
                rho_pred = math.exp(
                    coef_r["intercept"] +
                    coef_r["log_repeat"] * L +
                    coef_r.get("log_driveAmp", 0.0) * ld
                )
            rho_mode = "strong_regression"
        else:
            rho_pred = None
            rho_mode = "weak"
    out = {
        "driveAmp": drive,
        "repeat": repeat,
        "e_load_pred": e_load_pred,
        "u2_p99_pred": u2_pred,
        "rho_delta_rel_phase_pred": rho_pred,
        "rho_pred_mode": rho_mode,
        "rho_strong_prob": rho_strong_prob,
        "rho_pred_ge_0p05": rho_pred_ge,
        "off_step": off_step,
        "model_path": str(model_path),
        "model_sha1": sha1_file(model_path),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
