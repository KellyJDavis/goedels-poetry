import Mathlib.Tactic

/-!
# International Mathematical Olympiad 1998, Problem 4

Determine all pairs (a, b) of positive integers such that ab^2 + b + 7 divides a^2b + a + b.
-/

namespace Imo1998P4

/- determine -/ abbrev solution_set : Set (ℕ × ℕ) := sorry

theorem imo1998_p4 (a b : ℕ) :
    (0 < a ∧ 0 < b ∧ a * b^2 + b + 7 ∣ a^2 * b + a + b) ↔
    (a, b) ∈ solution_set := sorry


end Imo1998P4
