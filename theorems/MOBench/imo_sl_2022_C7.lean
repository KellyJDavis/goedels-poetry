
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2022 C7

Let $m$ be a positive integer and consider an arbitrary subset $S \subseteq \mathbb{Z}^m$.
We say that $S$ is *add-sup closed* if for any $v, w \in S$, their pointwise sum $v+w$ and
their pointwise maximum (or sup) $v \lor w$ are also in $S$.
A set $G \subseteq \mathbb{Z}^m$ is called an *add-sup generator* if the only add-sup
closed set containing $G$ is $\mathbb{Z}^m$ itself.

Find the smallest possible size of an add-sup generator, in terms of $m$.
-/
/- special open -/ open Finset Classical
class IsAddSupClosed {m : ℕ} (S : Set (Fin m → ℤ)) : Prop where
  add_mem : ∀ {v w}, v ∈ S → w ∈ S → v + w ∈ S
  sup_mem : ∀ {v w}, v ∈ S → w ∈ S → v ⊔ w ∈ S

def IsAddSupGenerator {m : ℕ} (G : Finset (Fin m → ℤ)) : Prop :=
  ∀ S : Set (Fin m → ℤ), ↑G ⊆ S → IsAddSupClosed S → S = Set.univ

def IsGoodSize (m n : ℕ) : Prop :=
  ∃ G : Finset (Fin m → ℤ), G.card ≤ n ∧ IsAddSupGenerator G

theorem imo_sl_2022_C7 (m n : ℕ) :
  IsGoodSize m n ↔ n ≥ (if m = 0 then 1 else if m ≤ 2 then 2 else 3) := by sorry
