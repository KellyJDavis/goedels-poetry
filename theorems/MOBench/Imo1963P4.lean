
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1963, Problem 4

Find all solutions x₁,x₂,x₃,x₄,x₅ of the system

  x₅ + x₂ = yx₁
  x₁ + x₃ = yx₂
  x₂ + x₄ = yx₃
  x₃ + x₅ = yx₄
  x₄ + x₁ = yx₅

where y is a parameter.
-/
abbrev SolutionSet (y : ℝ) : Set (ℝ × ℝ × ℝ × ℝ × ℝ) :=
  if y = 2 then
    -- When y = 2, all variables are equal: xᵢ = s for any s
    {p | ∃ s : ℝ, p = (s, s, s, s, s)}
  else if y * y + y - 1 = 0 then
    -- When y² + y - 1 = 0, we can have two free parameters
    {p | ∃ (s t : ℝ),
         p = (s, t, -s + y*t, -y*s - y*t, y*s - t)}
  else
    -- Otherwise, either all xᵢ = 0 or the system has no solution
    {(0, 0, 0, 0, 0)}

theorem imo1963_p4 (x₁ x₂ x₃ x₄ x₅ y : ℝ) :
    (x₁, x₂, x₃, x₄, x₅) ∈ SolutionSet y ↔
    (x₅ + x₂ = y * x₁ ∧
     x₁ + x₃ = y * x₂ ∧
     x₂ + x₄ = y * x₃ ∧
     x₃ + x₅ = y * x₄ ∧
     x₄ + x₁ = y * x₅) := by sorry
