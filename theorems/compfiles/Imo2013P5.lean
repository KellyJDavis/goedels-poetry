import Mathlib.Algebra.GeomSum
import Mathlib.Data.Real.Archimedean
import Mathlib.Tactic

/-!
# International Mathematical Olympiad 2013, Problem 5

Let ℚ>₀ be the set of positive rational numbers. Let f: ℚ>₀ → ℝ be a function satisfying
the conditions

  (1) f(x) * f(y) ≥ f(x * y)
  (2) f(x + y)    ≥ f(x) + f(y)

for all x,y ∈ ℚ>₀. Given that f(a) = a for some rational a > 1, prove that f(x) = x for
all x ∈ ℚ>₀.

-/

namespace Imo2013P5

theorem imo2013_p5
    (f : ℚ → ℝ)
    (H1 : ∀ x y, 0 < x → 0 < y → f (x * y) ≤ f x * f y)
    (H2 : ∀ x y, 0 < x → 0 < y → f x + f y ≤ f (x + y))
    (H_fixed_point : ∃ a, 1 < a ∧ f a = a) :
    ∀ x, 0 < x → f x = x := sorry


end Imo2013P5
