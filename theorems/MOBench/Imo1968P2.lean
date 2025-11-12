
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1968, Problem 2

Determine the set of natural numbers x such that
the sum of the decimal digits of x is equal to x² - 10x - 22.
-/
abbrev solution_set : Set ℕ := { 12 }

theorem imo1968_p2 (x : ℕ) :
    x ∈ solution_set ↔
    x^2 = 10 * x + 22 + (Nat.digits 10 x).prod := by sorry
