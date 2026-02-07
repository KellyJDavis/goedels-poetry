The proof (Round {{ prev_round_num }}) is not correct. Following is the compilation error message, where we use <error></error> to signal the position of the error.

{{ error_message_for_prev_round }}

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed analysis of the error message.

IMPORTANT: When you've completed your detailed analysis of the error message you should output your proof in Lean 4 code within a ```lean4...``` code block.

IMPORTANT: Your proof should actually prove the theorem or lemma using Lean 4 code. It should not contain sorry, admit, or any other Lean 4 tactics that indicate an incomplete proof.

IMPORTANT: Your corrected proof should contain ONLY the theorem/def and its proof. Do NOT include any import statements or preamble (like `import Mathlib`, `open`, `set_option`, `noncomputable section`, etc.). Start directly with your theorem/definition/lemma declaration.

IMPORTANT: Preserve the declaration header exactly as given (including whether it is `theorem` vs `lemma`, the declaration name, binders, and the statement type). Only change the proof body.
