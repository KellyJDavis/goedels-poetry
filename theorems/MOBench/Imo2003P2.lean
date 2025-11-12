
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2003, Problem 2

Determine all pairs of positive integers (a,b) such that

                  a²/(2ab² - b³ + 1)

is a positive integer.
-/
abbrev solution_set : Set (ℤ × ℤ) :=
  {p | ∃ k : ℤ, 0 < k ∧ (p = (2*k, 1) ∨ p = (k, 2*k) ∨ p = (8*k^4 - k, 2*k))}

theorem imo2003_p2 (a b : ℤ) :
    (a,b) ∈ solution_set ↔
    0 < a ∧ a < b ∧
    ∃ c, 0 < c ∧ c * (2 * a * b^2 - b^3 + 1) = a^2 := by sorry
