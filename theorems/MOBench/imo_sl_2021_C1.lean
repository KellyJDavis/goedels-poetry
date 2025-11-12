
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2021 C1

Consider a complete graph with an infinite vertex set $V$.
Each edge $xy$ is coloured such that for each vertex $v$, there exists only
  finitely many colours assigned to an edge incident with $v$.
Prove that if some of the edges has distinct colours, then there exists
  $x, y, z ∈ V$, pairwise distinct, such that $c_{xy} = c_{xz} ≠ c_{yz}$.
-/
structure FiniteIncidenceColouring (V α : Type*) where
  colour : V → V → α
  colour_symm (x y : V) : colour x y = colour y x
  incidence_finite (v : V) : Finite (Set.range (colour v))

variable [Infinite V] (C : FiniteIncidenceColouring V α)

theorem imo_sl_2021_C1 (h : ∀ c : α, ∃ x y : V, x ≠ y ∧ C.colour x y ≠ c) :
    ∃ x y z, y ≠ z ∧ C.colour x y = C.colour x z ∧ C.colour y z ≠ C.colour x z := by sorry
