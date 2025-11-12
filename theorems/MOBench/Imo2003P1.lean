
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2003, Problem 1

Let A be a 101-element subset of S = {1,2,...10⁶}. Prove that
there exist numbers t₁, t₂, ..., t₁₀₀ in S such that the sets

     Aⱼ = {x + tⱼ | x ∈ A},     j = 1,2, ..., 100

are pairwise disjoint.
-/
abbrev S : Finset ℕ := Finset.Icc 1 (10^6)

theorem imo2003_p1 (A : Finset ℕ) (hA : A ⊆ S) :
    ∃ t : Fin 100 → S,
      ∀ i j, i ≠ j → Disjoint {x + t i | x ∈ A} {x + t j | x ∈ A} := by sorry
