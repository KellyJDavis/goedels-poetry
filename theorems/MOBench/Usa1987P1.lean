
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# USA Mathematical Olympiad 1987, Problem 1

Determine all solutions to

     (m² + n)(m + n²) = (m - n)³

where m and n are non-zero integers.
-/
abbrev solution_set : Set (ℤ × ℤ) :=
  { (-1, -1), (8, -10), (9, -6), (9, -21) }

theorem usa1987_p1 (m n : ℤ) :
    (m, n) ∈ solution_set ↔
    m ≠ 0 ∧ n ≠ 0 ∧ (m^2 + n) * (m + n^2) = (m - n)^3 := by sorry
