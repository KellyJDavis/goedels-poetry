import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Order.Rearrangement
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Order.Interval.Finset.Nat

/-!
# International Mathematical Olympiad 1975, Problem 1

Let `x₁, x₂, ... , xₙ` and `y₁, y₂, ... , yₙ` be two sequences of real numbers, such that
`x₁ ≥ x₂ ≥ ... ≥ xₙ` and `y₁ ≥ y₂ ≥ ... ≥ yₙ`. Prove that if `z₁, z₂, ... , zₙ` is any permutation
of `y₁, y₂, ... , yₙ`, then `∑ (xᵢ - yᵢ)^2 ≤ ∑ (xᵢ - zᵢ)^2`
-/

namespace Imo1975P1

/- Let `n` be a natural number, `x` and `y` be as in the problem statement and `σ` be the
permutation of natural numbers such that `z = y ∘ σ` -/
variable (n : ℕ) (σ : Equiv.Perm ℕ) (x y : ℕ → ℝ)

theorem imo1975_p1
    (hx : AntitoneOn x (Finset.Icc 1 n)) (hy : AntitoneOn y (Finset.Icc 1 n))
    (hσ : {x | σ x ≠ x} ⊆ Finset.Icc 1 n) :
    ∑ i ∈ Finset.Icc 1 n, (x i - y i) ^ 2 ≤ ∑ i ∈ Finset.Icc 1 n, (x i - y (σ i)) ^ 2 := sorry


end Imo1975P1
