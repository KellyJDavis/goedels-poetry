
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2009 C2

For each $n ∈ ℕ$, find the largest integer $k$ such that the following holds:
  there exists injective functions $a_1, a_2, a_3 : [k] → ℕ$ such that
  $a_1(i) + a_2(i) + a_3(i) = n$ for all $i ∈ [k]$.
-/
/- special open -/ open Finset
structure GoodTripleFn (n : ℕ) (ι : Type*) where
  toFun : Fin 3 → ι → ℕ
  toFun_inj : ∀ j, (toFun j).Injective
  toFun_sum : ∀ i, ∑ j : Fin 3, toFun j i = n

/-- Final solution -/
theorem imo_sl_2009_C2 [Fintype ι] :
    Nonempty (GoodTripleFn n ι) ↔ Fintype.card ι ≤ 2 * n / 3 + 1 := by sorry
