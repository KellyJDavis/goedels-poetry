
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1994, Problem 4

Determine all ordered pairs of positive integers (m, n) such that

            (n³ + 1) / (mn - 1)

is an integer.
-/
abbrev SolutionSet : Set (ℤ × ℤ) :=
  {(1, 2), (1, 3), (2, 1), (2, 2), (2, 5), (3, 1), (3, 5), (5, 2), (5, 3)}

theorem imo1994_p4 (m n : ℤ) :
    (m, n) ∈ SolutionSet ↔
    0 < m ∧ 0 < n ∧ (m * n - 1) ∣ (n^3 + 1) := by sorry
