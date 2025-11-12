
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2022, Problem 3

Let k be a positive integer and let S be a finite set of odd prime
integers. Prove that there is at most one way (up to rotation and reflection)
to place the elements of S around a circle such that the product of any
two neighbors is of the form x² + x + k for some positive integer x.

-/
/- special open -/ open Finset







/-- The condition of the problem on a placement of numbers round a circle. -/
def Condition (k : ℕ) (S : Finset ℕ) (p : Fin (Finset.card S) → S) : Prop :=
  ∀ i, have : NeZero (Finset.card S) := ⟨i.pos.ne'⟩
  ∃ x : ℕ, 0 < x ∧ ((p i : ℕ) * (p (i + 1) : ℕ)) = x ^ 2 + x + k

theorem imo2023_p3
    {k : ℕ} (hk : 0 < k) (S : Finset ℕ) (hS : ∀ p ∈ S, Odd p ∧ Nat.Prime p)
    {p₁ p₂ : Fin (Finset.card S) → S} (hp₁ : Condition k S p₁) (hp₂ : Condition k S p₂) :
    (∃ i, ∀ j, p₂ j = p₁ (j + i)) ∨ ∃ i, ∀ j, p₂ j = p₁ (Fin.rev j + i) := by sorry
