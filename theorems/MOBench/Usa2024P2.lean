
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# USA Mathematical Olympiad 2024, Problem 2

Let S₁, S₂, ..., Sₙ be finite sets of integers whose intersection
is not empty. For each non-empty T ⊆ {S₁, S₂, ..., Sₙ}, the size of
the intersection of the sets in T is a multiple of the number of
sets in T. What is the least possible number of elements that are in
at least 50 sets?
-/
abbrev solution : ℕ := 50 * Nat.choose 100 50

structure Good (S : Fin 100 → Set ℤ) : Prop where
  finite : ∀ i, (S i).Finite
  nonempty_inter : ⋂ i, S i ≠ ∅
  card : ∀ T : Finset (Fin 100), T.Nonempty →
                 ∃ k : ℕ, (⋂ i ∈ T, S i).ncard * k = T.card

-- z is in at least k of the sets S.
abbrev InAtLeastKSubsets (S : Fin 100 → Set ℤ) (k : ℕ) (z : ℤ) : Prop :=
  k ≤ {i : Fin 100 | z ∈ S i }.ncard

theorem usa2024_p2 (n : ℕ) :
    IsLeast
      { k | ∃ S, Good S ∧
             k = {z : ℤ | InAtLeastKSubsets S k z }.ncard } solution := by sorry
