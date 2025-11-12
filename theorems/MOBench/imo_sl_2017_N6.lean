
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2017 N6

A multiset $S$ of positive rational numbers is called *nice* if
  both $\sum_{q ∈ S} q$ and $\sum_{q ∈ S} 1/q$ are integers.
Find all $n ∈ ℕ$ such that there exists infinitely many nice multisets $S$ of size $n$.
-/
/- special open -/ open Multiset
structure nice (S : Multiset ℚ) : Prop where
  pos : ∀ q : ℚ, q ∈ S → 0 < q
  sum_eq_int : ∃ k : ℤ, S.sum = k
  sum_inv_eq_int : ∃ k : ℤ, (S.map (·⁻¹)).sum = k

def good (n : ℕ) := {S : Multiset ℚ | card S = n ∧ nice S}.Infinite


theorem imo_sl_2017_N6 : good n ↔ 3 ≤ n := by sorry
