
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1998, Problem 4

Determine all pairs (a, b) of positive integers such that ab^2 + b + 7 divides a^2b + a + b.
-/
abbrev solution_set : Set (ℕ × ℕ) := {(11, 1), (49, 1)} ∪
  {(x,y) | ∃ k : ℕ , (x = 7 * k^2 ∧ y = 7 * k)}

theorem imo1998_p4 (a b : ℕ) :
    (0 < a ∧ 0 < b ∧ a * b^2 + b + 7 ∣ a^2 * b + a + b) ↔
    (a, b) ∈ solution_set := by sorry
