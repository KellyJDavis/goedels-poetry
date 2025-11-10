import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Order.Positive.Field
import Mathlib.Tactic

/-!
# USA Mathematical Olympiad 2023, Problem 2

Let ℝ+ be the set of positive real numbers.
Find all functions f: ℝ+ → ℝ+ that satisfy the equation

  f(x⬝y + f(x)) = x⬝f(y) + 2

for all x,y ∈ ℝ+.
-/

namespace Usa2023P2

abbrev PosReal : Type := { x : ℝ // 0 < x }

notation "ℝ+" => PosReal

/- determine -/ abbrev solution_set : Set (ℝ+ → ℝ+) := sorry

theorem usa2023_p2 (f : ℝ+ → ℝ+) :
    f ∈ solution_set ↔
    ∀ x y, f (x * y + (f x)) = x * (f y) + ⟨2, two_pos⟩ := sorry


end Usa2023P2
