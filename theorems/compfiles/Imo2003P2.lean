import Mathlib.Tactic

/-!
# International Mathematical Olympiad 2003, Problem 2

Determine all pairs of positive integers (a,b) such that

                  a²/(2ab² - b³ + 1)

is a positive integer.
-/

namespace Imo2003P2

/- determine -/ abbrev solution_set : Set (ℤ × ℤ) := sorry

theorem imo2003_p2 (a b : ℤ) :
    (a,b) ∈ solution_set ↔
    0 < a ∧ a < b ∧
    ∃ c, 0 < c ∧ c * (2 * a * b^2 - b^3 + 1) = a^2 := sorry


end Imo2003P2
