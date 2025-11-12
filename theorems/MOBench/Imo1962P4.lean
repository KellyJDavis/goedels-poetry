
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematics Olympiad 1962, Problem 4

Solve the equation
     cos² x + cos² (2 * x) + cos² (3 * x) = 1.
-/
def ProblemEquation (x : ℝ) : Prop :=
  cos x ^ 2 + cos (2 * x) ^ 2 + cos (3 * x) ^ 2 = 1

abbrev solutionSet : Set ℝ :=
  {x : ℝ | ∃ k : ℤ, x = (2 * ↑k + 1) * Real.pi / 4 ∨ x = (2 * ↑k + 1) * Real.pi / 6}

theorem imo1962_p4 {x : ℝ} : ProblemEquation x ↔ x ∈ solutionSet := by sorry
