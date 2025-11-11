import Mathlib.Tactic

/-!
# International Mathematical Olympiad 1965, Problem 2

Suppose that
  a₁₁x₁ + a₁₂x₂ + a₁₃x₃ = 0
  a₂₁x₁ + a₂₂x₂ + a₂₃x₃ = 0
  a₃₁x₁ + a₃₂x₂ + a₃₃x₃ = 0

where
 (A) a₁₁, a₂₂, a₃₃ are positive
 (B) the remaining aᵢⱼ are negative
 (C) in each row i, the sum of the coefficients aᵢⱼ is positive.

Prove that x₁ = x₂ = x₃ = 0.
-/

namespace Imo1965P2

theorem imo1965_p2 (x : Fin 3 → ℝ) (a : Fin 3 → Fin 3 → ℝ)
    (heqs : ∀ i, ∑ j : Fin 3, (a i j * x j) = 0)
    (hab : ∀ i j, if i = j then 0 < a i j else a i j < 0)
    (hc : ∀ i, 0 < ∑ j : Fin 3, a i j)
    : ∀ i, x i = 0 := sorry


end Imo1965P2
