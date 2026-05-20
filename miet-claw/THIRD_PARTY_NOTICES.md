# Third-party runtime dependencies

This repository contains the mietclaw agent/orchestration code. It does **not** vendor local native simulation engine checkouts or compiled binaries.

Expected external tools are configured through `config/local-agent.json` or environment variables:

- LAMMPS / MPI / Conda runtime
- MISA-KMC / CrystalKMC binary
- MoRe case directory and EAM potential files
- Optional KMC bridge script
- Optional OVITO Python runtime

Before publishing or redistributing third-party simulation engines, potential files, or compiled libraries, verify their own licenses and redistribution terms separately.
