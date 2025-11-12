
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# British Mathematical Olympiad 2024, Round 1, Problem 2

The sequence of integers a₀, a₁, ⋯ has the property that for each
i ≥ 2, aᵢ is either 2 * aᵢ₋₁ - aᵢ₋₂, or 2 * aᵢ₋₂ - aᵢ₋₁.

Given that a₂₀₂₃ and a₂₀₂₄ are consecutive integers, prove that a₀
and a₁ are consecutive.
-/
theorem uk2024_r1_p2 (a : ℕ → ℤ)
    (ha : ∀ i ≥ 2, a i = 2 * a (i - 1) - a (i - 2) ∨ a i = 2 * a (i - 2) - a (i - 1))
    (ha' : |a 2023 - a 2024| = 1) :
    |a 0 - a 1| = 1 := by sorry
