import Mathlib.Tactic

/-!
# International Mathematical Olympiad 2021, Problem 2

Let n be a natural number, and let x₁, ..., xₙ be real numbers.
Show that

     ∑ᵢ∑ⱼ √|xᵢ - xⱼ| ≤ ∑ᵢ∑ⱼ √|xᵢ + xⱼ|.

-/

namespace Imo2021P2

theorem imo2021_p2 (n : ℕ) (x : Fin n → ℝ) :
    ∑ i, ∑ j, √|x i - x j| ≤ ∑ i, ∑ j, √|x i + x j| := sorry


end Imo2021P2
