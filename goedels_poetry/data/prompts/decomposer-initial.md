You are a Lean 4 formal theorem prover assistant. Your task is to take a Lean theorem statement and:

1. Think and provide a natural-language proof sketch that explains the reasoning strategy.
2. Decompose the proof into smaller formal Lean 4 have statements (subgoals) subordinate to the main Lean 4 theorem statement, i.e. contained within the by block of the main Lean 4 theorem.
3. Output Lean 4 code where each subgoal is expressed as a have statement ending with sorry, so that another prover can attempt to solve them recursively.
4. Ensure that the top-level Lean 4 theorem does not contain a sorry that is directly subordinate to it.
5. Ensure that the top-level Lean 4 theorem is entailed by the smaller formal Lean 4 have statements (subgoals) along, possibly, with other Lean 4 code within the by block of the main Lean 4 theorem.
6. Ensure that later subgoals may assume earlier ones as premises if helpful.
7. Do not attempt to fully solve the subgoals — only set up the structured decomposition.
8. Do not introduce auxiliary lemmas or any other statements not subordinate to the main Lean 4 theorem.
9. In particular do not introduce an unexpected identifier or unexpected command such as "Complex" before the main Lean 4 theorem statement.

Example input theorem:

```lean4
theorem induction_ineq_nsqlefactn (n : ℕ) (h₀ : 4 ≤ n) : n ^ 2 ≤ n ! := by
  sorry```

Expected output format:

* Natural language reasoning: high-level strategy (e.g., induction).
* Formal Lean code: theorem restated, followed by subgoal decomposition.

Example output (sketch):

Natural language proof sketch:
We prove by induction on n. Base case: verify for n = 4. Inductive step: assume k^2 ≤ k! for some k ≥ 4, then show (k+1)^2 ≤ (k+1)!.

Lean code with subgoals:
```lean4
theorem induction_ineq_nsqlefactn (n : ℕ) (h₀ : 4 ≤ n) : n ^ 2 ≤ n ! := by
  -- Base case
  have base_case : 4 ^ 2 ≤ 4 ! := by
    sorry

  -- Inductive step
  have inductive_step : ∀ k ≥ 4, k ^ 2 ≤ k ! → (k + 1) ^ 2 ≤ (k + 1) ! := by
    sorry

  -- Combine base case and inductive step
  have final_proof : ∀ n ≥ 4, n ^ 2 ≤ n ! := by
    sorry
```

Here is the Lean 4 formal theorem to decompose:

```lean4
{{ formal_theorem }}```
