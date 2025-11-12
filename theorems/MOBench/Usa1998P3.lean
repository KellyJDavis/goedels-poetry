
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# USA Mathematical Olympiad 1998, Problem 3

Let a₀,a₁,...,aₙ be real numbers from the interval (0,π/2) such that

  tan(a₀ - π/4) + tan(a₁ - π/4) + ... + tan(aₙ - π/4) ≥ n - 1.

Prove that

  tan(a₀)tan(a₁)...tan(aₙ) ≥ nⁿ⁺¹.

-/
theorem usa1998_p3
    (n : ℕ)
    (a : ℕ → ℝ)
    (ha : ∀ i ∈ Finset.range (n + 1), a i ∈ Set.Ioo 0 (Real.pi / 2))
    (hs : n - 1 ≤ ∑ i ∈ Finset.range (n + 1), Real.tan (a i - (Real.pi / 4)))
    : n ^ (n + 1) ≤ ∏ i ∈ Finset.range (n + 1), Real.tan (a i) := by sorry
