
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2023 N3

For any positive integer $n$ and $k ≥ 2$, define $ν_k(n)$
  as the largest exponent $r$ such that $k^r ∣ n$.
Prove the following:
1. there are infinitely many $n$ such that $ν_{10}(n!) > ν_9(n!)$; and
2. there are infinitely many $n$ such that $ν_{10}(n!) < ν_9(n!)$.
-/
theorem imo_sl_2023_N3 (N : ℕ) :
    (∃ n > N, padicValNat 9 n.factorial < padicValNat 10 n.factorial) ∧
    (∃ n > N, padicValNat 10 n.factorial < padicValNat 9 n.factorial) := by sorry
