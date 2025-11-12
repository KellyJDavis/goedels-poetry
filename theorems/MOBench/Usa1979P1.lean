
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# USA Mathematical Olympiad 1979, Problem 1

Determine all non-negative integral solutions $(n_1,n_2,\dots , n_{14})$ if any,
apart from permutations, of the Diophantine Equation $n_1^4+n_2^4+\cdots +n_{14}^4=1599$.
-/
/--
A type representing assignments to the variables $n_1$, $n_2$, ..., $n_{14}$,
quotiented by permutations of indices.
-/
structure MultisetNatOfLen14 where
  s : Multiset ℕ
  p : Multiset.card s = 14

abbrev SolutionSet : Set MultisetNatOfLen14 := ∅

theorem usa1979_p1 : ∀ e, e ∈ SolutionSet ↔ (e.s.map (fun x ↦ x ^ 4)).sum = 1599 := by sorry
