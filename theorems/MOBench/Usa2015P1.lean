
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# USA Mathematical Olympiad 2015, Problem 1

Solve in integers the equation x² + xy + y² = ((x + y) / 3 + 1)³.
-/
abbrev SolutionSet : Set (ℤ × ℤ) :=
  {⟨x, y⟩ | ∃ n, x = n ^ 3 + 3 * n ^ 2 - 1 ∧ y = -n ^ 3 + 3 * n + 1} ∪
  {⟨x, y⟩ | ∃ n, y = n ^ 3 + 3 * n ^ 2 - 1 ∧ x = -n ^ 3 + 3 * n + 1}

theorem usa2015_p1 (x y : ℤ) :
    ⟨x, y⟩ ∈ SolutionSet ↔
    x^2 + x * y + y^2 = ((x + y) / (3 : ℚ) + 1)^3 := by sorry
