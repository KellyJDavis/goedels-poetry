import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith

/-!
# International Mathematical Olympiad 2011, Problem 3

Let f : ℝ → ℝ be a function that satisfies

   f(x + y) ≤ y * f(x) + f(f(x))

for all x and y. Prove that f(x) = 0 for all x ≤ 0.
-/

namespace Imo2011P3

theorem imo2011_p3 (f : ℝ → ℝ) (hf : ∀ x y, f (x + y) ≤ y * f x + f (f x)) :
    ∀ x ≤ 0, f x = 0 := sorry


end Imo2011P3
