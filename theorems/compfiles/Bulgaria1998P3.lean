import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset
import Mathlib.Analysis.SpecificLimits.Basic

/-!
# Bulgarian Mathematical Olympiad 1998, Problem 3

Let ℝ⁺ be the set of positive real numbers. Prove that there does not exist a function
f: ℝ⁺ → ℝ⁺ such that

                (f(x))² ≥ f(x + y) * (f(x) + y)

for every x,y ∈ ℝ⁺.

-/

namespace Bulgaria1998P3

theorem bulgaria1998_p3
    (f : ℝ → ℝ)
    (hpos : ∀ x : ℝ, 0 < x → 0 < f x)
    (hf : (∀ x y : ℝ, 0 < x → 0 < y → (f (x + y)) * (f x + y) ≤ (f x)^2)) :
    False := sorry


end Bulgaria1998P3
