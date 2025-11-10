import Mathlib.Tactic

/-!
# International Mathematical Olympiad 1961, Problem 1.

Given constants a and b, solve the system of equations

             x + y + z = a
          x² + y² + z² = b²
                    xy = z²

for x,y,z. Give the conditions that a and b must satisfy so that
the solutions x,y,z are distinct positive numbers.
-/

namespace Imo1961P1

abbrev IsSolution (a b x y z : ℝ) : Prop :=
    x + y + z = a ∧
    x^2 + y^2 + z^2 = b^2 ∧
    x * y = z^2

/- determine -/ abbrev xyz_of_ab (a b : ℝ) : Set (ℝ × ℝ × ℝ) := sorry

/- determine -/ abbrev ab_that_make_xyz_positive_distinct : Set (ℝ × ℝ) := sorry

theorem imo1961_p1a (a b x y z : ℝ) :
    ⟨x,y,z⟩ ∈ xyz_of_ab a b ↔ IsSolution a b x y z := sorry

theorem imo1961_p1b (a b : ℝ) :
    ⟨a,b⟩ ∈ ab_that_make_xyz_positive_distinct ↔
      (∀ x y z, IsSolution a b x y z → 0 < x ∧ 0 < y ∧ 0 < z ∧ [x,y,z].Nodup) := sorry



end Imo1961P1
