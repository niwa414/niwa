#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def relpath(from_dir: Path, target: Path) -> str:
    return Path(os.path.relpath(str(target), str(from_dir))).as_posix()


def artifact_from_passfail(passfail: dict, suffix: str) -> str:
    for item in passfail.get("artifacts", []):
        if isinstance(item, str) and item.endswith(suffix):
            return item
    return ""


def load_case(repo_root: Path, case_id: str) -> dict:
    pf_path = repo_root / "outputs" / case_id / "analysis" / "PASSFAIL.json"
    pf = read_json(pf_path)
    metrics = pf.get("metrics", {}) if isinstance(pf.get("metrics"), dict) else {}
    mp4_rel = artifact_from_passfail(pf, ".mp4")
    gif_rel = artifact_from_passfail(pf, ".gif")
    mp4_path = (repo_root / mp4_rel).resolve() if mp4_rel else Path()
    gif_path = (repo_root / gif_rel).resolve() if gif_rel else Path()
    return {
        "case_id": case_id,
        "passfail_path": pf_path,
        "status": str(pf.get("status") or pf.get("result") or "MISSING").upper(),
        "metrics": metrics,
        "mp4_path": mp4_path,
        "gif_path": gif_path,
    }


def write_play_script(path: Path, index_path: Path, mp4_paths: list[Path]) -> None:
    script_dir = path.parent
    index_rel = relpath(script_dir, index_path)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"",
        f"INDEX=\"$SCRIPT_DIR/{index_rel}\"",
        "",
        "open_file() {",
        "  local f=\"$1\"",
        "  if command -v open >/dev/null 2>&1; then",
        "    open \"$f\"",
        "  elif command -v xdg-open >/dev/null 2>&1; then",
        "    xdg-open \"$f\" >/dev/null 2>&1 &",
        "  else",
        "    echo \"Open manually: $f\"",
        "  fi",
        "}",
        "",
        "open_file \"$INDEX\"",
    ]
    for mp4 in mp4_paths:
        mp4_rel = relpath(script_dir, mp4)
        lines.append(f"open_file \"$SCRIPT_DIR/{mp4_rel}\"")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def render_index(
    out_path: Path,
    out_dir: Path,
    cases: list[dict],
    suite_passfail_path: Path,
    mapping_path: Path,
    evidence_pack_index: Path,
) -> None:
    generated = datetime.now(timezone.utc).isoformat()
    case_titles = {
        "m29-hf3d-1-merge-compression-recapture": "HF3D-1 Merge + Compression + Recapture",
        "m29-hf3d-2-formation-microinstability": "HF3D-2 Formation Micro-instability",
        "m29-hf3d-3-engineering-diffusion-load": "HF3D-3 Engineering Diffusion + Load",
    }

    head = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HF3D Demo Pack</title>
  <style>
    :root {{
      --bg: #f4f3ef;
      --panel: #fffdf9;
      --ink: #1e2329;
      --ink-soft: #4a5663;
      --line: #d8d3c6;
      --accent: #2e5b7a;
      --ok: #1f8b4c;
      --bad: #b8332a;
    }}
    body {{
      margin: 0;
      background: linear-gradient(145deg, #f4f3ef 0%, #e8ecef 55%, #f1eee6 100%);
      color: var(--ink);
      font-family: \"Avenir Next\", \"Helvetica Neue\", Helvetica, Arial, sans-serif;
    }}
    .wrap {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 24px;
    }}
    .hero {{
      background: radial-gradient(circle at 10% 10%, #ffffff 0%, #eef2f5 55%, #e4e2da 100%);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 18px 20px;
      box-shadow: 0 8px 22px rgba(20, 25, 33, 0.08);
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 8px 0;
      font-size: 28px;
      letter-spacing: 0.2px;
    }}
    .muted {{
      color: var(--ink-soft);
      margin: 0;
      font-size: 14px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 14px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px;
      box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
    }}
    .title {{
      margin: 0 0 8px 0;
      font-size: 18px;
      color: var(--accent);
    }}
    .status {{
      margin: 0 0 10px 0;
      font-size: 13px;
      color: var(--ink-soft);
    }}
    .ok {{
      color: var(--ok);
      font-weight: 600;
    }}
    .bad {{
      color: var(--bad);
      font-weight: 600;
    }}
    video {{
      width: 100%;
      border-radius: 8px;
      border: 1px solid #d4dbe3;
      background: #000;
    }}
    ul {{
      margin: 10px 0 0 16px;
      padding: 0;
      color: var(--ink-soft);
      font-size: 14px;
    }}
    li {{
      margin: 4px 0;
    }}
    a {{
      color: #0f4a7a;
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>Helion-style HF3D Demo Pack</h1>
      <p class="muted">Generated: {generated}</p>
      <p class="muted">Three high-fidelity 3D narratives with linked evidence and one-click playback support.</p>
    </div>
    <div class="grid">
"""

    cards = []
    for item in cases:
        cid = item["case_id"]
        title = case_titles.get(cid, cid)
        status = item["status"]
        cls = "ok" if status == "PASS" else "bad"
        metrics = item.get("metrics", {})
        frame_count = metrics.get("frames_rendered")
        mp4 = item["mp4_path"]
        gif = item["gif_path"]
        pf = item["passfail_path"]
        mp4_rel = relpath(out_dir, mp4) if mp4 else ""
        gif_rel = relpath(out_dir, gif) if gif else ""
        pf_rel = relpath(out_dir, pf)
        metrics_rel = relpath(out_dir, pf.parent / "metrics.json")
        summary_rel = relpath(out_dir, next((p for p in pf.parent.glob("hf3d_*.md")), pf.parent / "README.md"))
        cards.append(
            f"""      <section class="card">
        <h2 class="title">{title}</h2>
        <p class="status">Case: <code>{cid}</code> | Status: <span class="{cls}">{status}</span> | frames: <code>{frame_count}</code></p>
        <video controls preload="metadata" src="{mp4_rel}"></video>
        <ul>
          <li><a href="{mp4_rel}">MP4</a></li>
          <li><a href="{gif_rel}">GIF</a></li>
          <li><a href="{pf_rel}">PASSFAIL.json</a></li>
          <li><a href="{metrics_rel}">metrics.json</a></li>
          <li><a href="{summary_rel}">summary.md</a></li>
        </ul>
      </section>"""
        )

    suite_rel = relpath(out_dir, suite_passfail_path)
    map_rel = relpath(out_dir, mapping_path)
    evidence_rel = relpath(out_dir, evidence_pack_index)

    tail = f"""
    </div>
    <div class="hero" style="margin-top:16px">
      <h2 class="title" style="margin-top:0">Evidence Links</h2>
      <ul>
        <li><a href="{suite_rel}">HF3D suite gate PASSFAIL</a></li>
        <li><a href="{map_rel}">Helion mapping YAML</a></li>
        <li><a href="{evidence_rel}">Evidence pack index</a></li>
      </ul>
    </div>
  </div>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(head + "\n".join(cards) + tail, encoding="utf-8")


def write_summary(path: Path, metrics: dict) -> None:
    lines = [
        "# HF3D Demo Pack",
        "",
        f"- all_hf3d_pass: `{metrics.get('all_hf3d_pass')}`",
        f"- all_mp4_exists: `{metrics.get('all_mp4_exists')}`",
        f"- demo_index_exists: `{metrics.get('demo_index_exists')}`",
        f"- playback_script_exists: `{metrics.get('playback_script_exists')}`",
        f"- evidence_links_written: `{metrics.get('evidence_links_written')}`",
        "",
        "This package provides a single-page browser view and one-click playback for the three HF3D narratives.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HF3D Helion-style demo pack with index + playback script.")
    parser.add_argument("--merge-case", default="m29-hf3d-1-merge-compression-recapture")
    parser.add_argument("--formation-case", default="m29-hf3d-2-formation-microinstability")
    parser.add_argument("--engineering-case", default="m29-hf3d-3-engineering-diffusion-load")
    parser.add_argument("--suite-case", default="m29-d1-hf3d-suite-gate")
    parser.add_argument("--index-out", required=True)
    parser.add_argument("--play-script-out", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    index_out = Path(args.index_out)
    play_out = Path(args.play_script_out)
    metrics_out = Path(args.metrics_out)
    summary_out = Path(args.summary_out)
    if not index_out.is_absolute():
        index_out = (repo_root / index_out).resolve()
    if not play_out.is_absolute():
        play_out = (repo_root / play_out).resolve()
    if not metrics_out.is_absolute():
        metrics_out = (repo_root / metrics_out).resolve()
    if not summary_out.is_absolute():
        summary_out = (repo_root / summary_out).resolve()

    merge = load_case(repo_root, args.merge_case)
    formation = load_case(repo_root, args.formation_case)
    engineering = load_case(repo_root, args.engineering_case)
    cases = [merge, formation, engineering]

    suite_passfail = repo_root / "outputs" / args.suite_case / "analysis" / "PASSFAIL.json"
    suite_data = read_json(suite_passfail)
    suite_status = str(suite_data.get("status") or suite_data.get("result") or "MISSING").upper()

    all_hf3d_pass = bool(
        suite_status == "PASS"
        and all(item["status"] == "PASS" for item in cases)
        and all(bool(item["metrics"].get("render_success")) for item in cases)
    )
    all_mp4_exists = bool(all(item["mp4_path"].exists() for item in cases))

    out_dir = index_out.parent
    mapping_path = repo_root / "evidence" / "pack" / "helion_mapping.yaml"
    evidence_pack_index = repo_root / "evidence" / "pack" / "index.md"
    render_index(
        out_path=index_out,
        out_dir=out_dir,
        cases=cases,
        suite_passfail_path=suite_passfail,
        mapping_path=mapping_path,
        evidence_pack_index=evidence_pack_index,
    )
    write_play_script(play_out, index_out, [item["mp4_path"] for item in cases])

    metrics = {
        "merge_case_id": args.merge_case,
        "formation_case_id": args.formation_case,
        "engineering_case_id": args.engineering_case,
        "suite_case_id": args.suite_case,
        "all_hf3d_pass": all_hf3d_pass,
        "all_mp4_exists": all_mp4_exists,
        "demo_index_exists": bool(index_out.exists() and index_out.stat().st_size > 0),
        "playback_script_exists": bool(play_out.exists() and play_out.stat().st_size > 0),
        "evidence_links_written": bool(mapping_path.exists() and evidence_pack_index.exists() and suite_passfail.exists()),
        "index_path": str(index_out),
        "play_script_path": str(play_out),
    }

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    write_summary(summary_out, metrics)


if __name__ == "__main__":
    main()
