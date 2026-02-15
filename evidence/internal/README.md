# Internal Parity Manifest

This directory holds optional internal-only evidence bindings used by:

- `tools/analyze_helion_internal_parity.py`

Use `helion_internal_parity_manifest.json` to bind:

- GPU runtime proof artifact
- Private shot/calibration dataset artifact
- Private hardware model validation artifact

Rules enforced by the analyzer:

- Every enabled artifact needs a file path and exact SHA-256 hash.
- GPU runtime proof regex must be backend-specific (`CUDA`/`HIP`/`ROCm`/`SYCL`).
- Disabled sections do not count as bound evidence.

Template:

- `helion_internal_parity_manifest.template.json`

Populate hashes with:

```bash
shasum -a 256 <file>
```
