import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Order.Positive.Field
import Mathlib.Tactic

/-!
# International Mathematical Olympiad 2022, Problem 2

Let ℝ+ be the set of positive real numbers.
Determine all functions f: ℝ+ → ℝ+ such that
for each x ∈ ℝ+, there is exactly one y ∈ ℝ+
satisfying

  x·f(y) + y·f(x) ≤ 2
-/

namespace Imo2022P2

abbrev PosReal : Type := { x : ℝ // 0 < x }

notation "ℝ+" => PosReal

/- determine -/ abbrev solution_set : Set (ℝ+ → ℝ+) := sorry

theorem imo2022_p2 (f : ℝ+ → ℝ+) :
    f ∈ solution_set ↔
    ∀ x, ∃! y, x * f y + y * f x ≤ ⟨2, two_pos⟩ := sorry


end Imo2022P2
