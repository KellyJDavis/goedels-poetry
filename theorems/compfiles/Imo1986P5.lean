import Mathlib.Data.NNReal.Basic

/-!
# International Mathematical Olympiad 1986, Problem 5

Find all functions `f`, defined on the non-negative real numbers and taking nonnegative real values,
such that:

- $f(xf(y))f(y) = f(x + y)$ for all $x, y \ge 0$,
- $f(2) = 0$,
- $f(x) \ne 0$ for $0 \le x < 2$.
-/

open scoped NNReal

namespace Imo1986P5

structure IsGood (f : ℝ≥0 → ℝ≥0) : Prop where
  map_add_rev x y : f (x * f y) * f y = f (x + y)
  map_two : f 2 = 0
  map_ne_zero : ∀ x < 2, f x ≠ 0

/- determine -/ abbrev SolutionSet : Set (ℝ≥0 → ℝ≥0) := sorry

theorem imo1986_p5 {f : ℝ≥0 → ℝ≥0} : IsGood f ↔ f ∈ SolutionSet := sorry


end Imo1986P5
