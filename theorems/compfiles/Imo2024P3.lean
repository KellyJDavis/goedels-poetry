import Mathlib.Tactic
import Mathlib.Data.Set.Card

/-!
# International Mathematical Olympiad 2024, Problem 3

Let a₁, a₂, a₃, ... be an infinite sequence of positive integers,
and let N be a positive integer. Suppose that, for each n > N,
aₙ is equal to the number of times aₙ₋₁ appears in the list
a₁, a₂, ..., aₙ₋₁.

Prove that at least one of the sequences a₁, a₃, a₅, ... and
a₂, a₄, a₆, ... is eventually periodic.
-/

namespace Imo2024P3

abbrev EventuallyPeriodic (b : ℕ → ℕ) : Prop :=
  ∃ p M, ∀ m ≥ M, b (m + p) = b m

theorem imo2024_p3
    (a : ℕ → ℕ)
    (apos : ∀ i, 0 < a i)
    (N : ℕ)
    (h : ∀ n ≥ N, a (n+1) = Set.ncard {i : ℕ | i ≤ n ∧ a i = a n})
    : EventuallyPeriodic (fun i ↦ a (1 + 2 * i)) ∨
      EventuallyPeriodic (fun i ↦ a (2 * i)) := sorry

end Imo2024P3
