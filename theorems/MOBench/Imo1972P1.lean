
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1972, Problem 1

Prove that from a set of ten distinct two-digit numbers (in
decimal), it is possible to select two disjoint subsets whose
members have the same sum.
-/
theorem imo1972_p1 (S : Finset ℕ)
    (Scard : S.card = 10)
    (Sdigits : ∀ n ∈ S, (Nat.digits 10 n).length = 2) :
    ∃ S1 S2 : Finset ℕ, S1 ⊆ S ∧ S2 ⊆ S ∧
       Disjoint S1 S2 ∧ ∑ n ∈ S1, n = ∑ n ∈ S2, n := by sorry
