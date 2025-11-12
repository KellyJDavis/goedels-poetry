
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2007 C3

Find all finite groups $G$ such that there exists a subset $S \subseteq G$ for which
 the number of triples $(x, y, z) \in S^3 \cup (G \setminus S)^3$ such that $xyz = 1$ is $2007$.
-/
/- special open -/ open Finset
variable [Fintype G] [DecidableEq G] [Group G]

def tripleSet (S : Finset G) : Finset (Fin 3 → G) :=
  (Fintype.piFinset fun _ ↦ S) ∪ (Fintype.piFinset fun _ ↦ Sᶜ)

def filtered_tripleSet (S : Finset G) : Finset (Fin 3 → G) :=
  (tripleSet S).filter fun p ↦ p 0 * p 1 * p 2 = 1

theorem imo_sl_2007_C3 :
  (∃ S : Finset G, (filtered_tripleSet S).card = 2007) ↔
  Fintype.card G = 69 ∨ Fintype.card G = 84 := by sorry
