# MOBench Problem Library

This directory contains Lean formalizations generated from the MOBench dataset bundled with the Goedel-Prover-V2 project.<sup>[1](https://github.com/Goedel-LM/Goedel-Prover-V2/blob/main/dataset/MOBench.jsonl)</sup>

- The Lean files were created by running the `lean-from-json.py` CLI utility in this directory against the published JSONL dataset.
- The collection targets Lean `v4.9.0`, matching the compiler version expected by the Goedel-Prover-V2 project.
- Each entry preserves the problem declarations so they can be imported directly or adapted for downstream automation workflows.

To rebuild or refresh the files, run:

```bash
python theorems/MOBench/lean-from-json.py convert --json dataset/MOBench.jsonl --output-dir theorems/MOBench
```

Adjust the `--json` path if the dataset lives elsewhere, and use `--overwrite` when syncing updates from the upstream source.
