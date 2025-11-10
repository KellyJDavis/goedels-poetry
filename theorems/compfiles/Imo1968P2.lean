import Mathlib.Data.Nat.Digits
import Mathlib.Tactic

/-!
# International Mathematical Olympiad 1968, Problem 2

Determine the set of natural numbers x such that
the sum of the decimal digits of x is equal to x² - 10x - 22.
-/

namespace Imo1968P2

/- determine -/ abbrev solution_set : Set ℕ := sorry

theorem imo1968_p2 (x : ℕ) :
    x ∈ solution_set ↔
    x^2 = 10 * x + 22 + (Nat.digits 10 x).prod := sorry


end Imo1968P2
