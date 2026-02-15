# Cases

Each case lives under `cases/<case_id>/` with:

- `case.json`: run/analyze commands, thresholds, and artifact list.
- `inputs/`: input files referenced by the case.

`tools/run_case.py` expands these placeholders in `case.json`:

- `{python}`: current Python interpreter
- `{repo_root}`: repository root
- `{case_dir}`: case directory
- `{output_root}`: `outputs/<case_id>`
- `{output_raw}`: `outputs/<case_id>/raw`
- `{output_analysis}`: `outputs/<case_id>/analysis`
- `{output_plots}`: `outputs/<case_id>/plots`
- `{output_logs}`: `outputs/<case_id>/logs`

Run a case:

```
python tools/run_case.py --case <case_id> --stage all --update-evidence
```
