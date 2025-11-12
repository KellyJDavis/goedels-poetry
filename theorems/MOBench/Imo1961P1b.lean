
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1961, Problem 1.

Given constants a and b, solve the system of equations

             x + y + z = a
          x² + y² + z² = b²
                    xy = z²

for x,y,z. Give the conditions that a and b must satisfy so that
the solutions x,y,z are distinct positive numbers.
-/
abbrev IsSolution (a b x y z : ℝ) : Prop :=
    x + y + z = a ∧
    x^2 + y^2 + z^2 = b^2 ∧
    x * y = z^2

abbrev ab_that_make_xyz_positive_distinct : Set (ℝ × ℝ) :=
  { q | let ⟨a,b⟩ := q
        b^2 < a^2 ∧ a^2 < 3 * b ^ 2 }

theorem imo1961_p1b (a b : ℝ) :
    ⟨a,b⟩ ∈ ab_that_make_xyz_positive_distinct ↔
      (∀ x y z, IsSolution a b x y z → 0 < x ∧ 0 < y ∧ 0 < z ∧ [x,y,z].Nodup) := by sorry
