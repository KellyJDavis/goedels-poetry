
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2023, Problem 3

For each integer k ≥ 2, determine all infinite sequences of positive
integers a₁, a₂, ... for which there exists a polynomial P of the form

  P(x) = xᵏ + cₖ₋₁xᵏ⁻¹ + ... + c₁x + c₀,

where c₀, c₁, ..., cₖ₋₁ are non-negative integers, such that

  P(aₙ) = aₙ₊₁aₙ₊₂⋯aₙ₊ₖ

for every integer n ≥ 1.
-/
abbrev SolutionSet {k : ℕ} (hk : 2 ≤ k) : Set (ℕ+ → ℕ+) :=
  {a | ∃ (a₁ m : ℕ), 0 < a₁ ∧ 0 ≤ m ∧ ∀ n : ℕ+, a n = a₁ + (n - 1) * m}

theorem imo2023_p3 {k : ℕ} (hk : 2 ≤ k) (a : ℕ+ → ℕ+) :
    a ∈ SolutionSet hk ↔
    (∃ P : Polynomial ℤ, P.Monic ∧ P.degree = k ∧
     (∀ n, n ≤ k → 0 ≤ P.coeff n) ∧
      ∀ n : ℕ+,
        P.eval ((a n) : ℤ) =
        ∏ i ∈ Finset.range k, a ⟨n + i + 1, Nat.succ_pos _⟩) := by sorry
