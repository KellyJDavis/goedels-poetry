
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1962, Problem 2

Determine all real numbers x which satisfy

  √(3 - x) - √(x + 1) > 1/2.
-/
abbrev SolutionSet : Set ℝ := Set.Ico (-1) (1 - √31 / 8)

theorem imo1962_p2 (x : ℝ) :
    x ∈ SolutionSet ↔
    x ≤ 3 ∧ -1 ≤ x ∧ 1/2 < √(3 - x) - √(x + 1) := by sorry
