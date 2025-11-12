
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2024, Problem 3

Let a₁, a₂, a₃, ... be an infinite sequence of positive integers,
and let N be a positive integer. Suppose that, for each n > N,
aₙ is equal to the number of times aₙ₋₁ appears in the list
a₁, a₂, ..., aₙ₋₁.

Prove that at least one of the sequences a₁, a₃, a₅, ... and
a₂, a₄, a₆, ... is eventually periodic.
-/
/- special open -/ open Finset






def Condition (a : ℕ → ℕ) (N : ℕ) : Prop :=
  (∀ i, 0 < a i) ∧ ∀ n, N < n → a n = Finset.card (filter (λ i => a i = a (n - 1)) (Finset.range n))

def EventuallyPeriodic (b : ℕ → ℕ) : Prop :=
  ∃ p M, 0 < p ∧ ∀ m, M ≤ m → b (m + p) = b m

theorem imo2024_p3 {a : ℕ → ℕ} {N : ℕ} (h : Condition a N) :
    EventuallyPeriodic (fun i ↦ a (2 * i)) ∨ EventuallyPeriodic (fun i ↦ a (2 * i + 1)) := by sorry
