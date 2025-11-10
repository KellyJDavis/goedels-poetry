import Mathlib.Data.Real.Basic
import Mathlib.Analysis.MeanInequalities

/-!
# Iranian Mathematical Olympiad 1998, problem 3

Let x₁, x₂, x₃, x₄ be positive real numbers such that

  x₁ ⬝ x₂ ⬝ x₃ ⬝ x₄ = 1.

Prove that
  x₁³ + x₂³ + x₃³ + x₄³ ≥ max(x₁ + x₂ + x₃ + x₄, 1/x₁ + 1/x₂ + 1/x₃ + 1/x₄).

-/

namespace Iran1998P3

theorem iran1998_p3
    (x : ℕ → ℝ)
    (x_positive : ∀ i, 0 < x i)
    (h : ∏ i ∈ Finset.range 4, x i = 1)
    : max (∑ i ∈ Finset.range 4, x i) (∑ i ∈ Finset.range 4, 1 / x i)
     ≤ ∑ i ∈ Finset.range 4, (x i)^(3 : ℝ) := sorry


end Iran1998P3
