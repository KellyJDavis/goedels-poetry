The proof sketch (Round {{ prev_round_num }}) is not correct. Following is the compilation error message, where we use <error></error> to signal the position of the error.

{{ error_message_for_prev_round }}

Before producing the Lean 4 code to sketch a proof to the given theorem, provide a detailed analysis of the error message.

IMPORTANT constraint (must follow):
- Never place a `sorry` inside a nested subproof (e.g. inside a `calc` step `... := by sorry`, inside `match`/`cases` branches, or inside any nested `by` block).
- If a subgoal is unproven, it must end with a single top-level placeholder:
  ```lean4
  have h_name : T := by
    sorry
  ```
  If you want to outline a structured proof (like `calc`), put the outline in comments, but keep the proof itself as `by sorry`.
