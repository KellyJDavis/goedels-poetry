# MiniF2F Problem Library

This directory contains Lean formalizations generated from the MiniF2F dataset bundled with the Goedel-Prover-V2 project.<sup>[1](https://github.com/Goedel-LM/Goedel-Prover-V2/blob/main/dataset/minif2f.jsonl)</sup>

- The Lean files were created by running the `lean-from-json.py` CLI utility in this directory against the published JSONL dataset.
- The collection targets Lean `v4.9.0`, matching the compiler version expected by the Goedel-Prover-V2 project.
- Each entry preserves the problem declarations so they can be imported directly or adapted for downstream automation workflows.

To rebuild or refresh the files, run:

```bash
python theorems/minif2f/lean-from-json.py convert --json dataset/minif2f.jsonl --output-dir theorems/minif2f
```

Adjust the `--json` path if the dataset lives elsewhere, and use `--overwrite` when syncing updates from the upstream source.
