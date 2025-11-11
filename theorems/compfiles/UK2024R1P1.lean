import Mathlib.Data.Set.Card
import Mathlib.Logic.Equiv.Defs
import Mathlib.Tactic

/-!
# British Mathematical Olympiad 2024, Round 1, Problem 1

An unreliable typist can guarantee that when they try to type a word with
different letters, every letter of the word will appear exactly once in what
they type, and each letter will occur at most one letter late (though it may
occur more than one letter early). Thus, when trying to type MATHS, the
typist may type MATHS, MTAHS or TMASH, but not ATMSH.

Determine, with proof, the number of possible spellings of OLYMPIADS
that might be typed.
-/

namespace UK2024R1P1

/- determine -/ abbrev solution_value : ℕ := sorry

/-
Since OLYMPIADS has no duplicate letters, then the set of spellings is just a
subset of the permutations of 9 elements.
-/
theorem uk2024_r1_p1 :
  {f : Equiv.Perm (Fin 9) | ∀ k, (f k : ℕ) ≤ k + 1}.ncard = solution_value := sorry


end UK2024R1P1
