
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2007 A6

Let $R$ be a totally ordered commutative ring and let $n \ge 5$ be an integer.
Prove that for any sequence $a_1, a_2, \dots, a_n \in R$, the following inequality holds:
$$ \left(3 \sum_{i = 1}^n a_i^2 a_{i + 1}\right)^2 \le 2 \left(\sum_{i = 1}^n a_i^2\right)^3 $$
where the sum is cyclic, i.e., $a_{n+1} = a_1$.
-/
/- special open -/ open Finset
theorem imo_sl_2007_A6 [LinearOrderedCommRing R] (n : ℕ) (hn : 5 ≤ n) (a : Fin n → R) :
  (3 * ∑ i, a i ^ 2 * a (finRotate n i)) ^ 2 ≤ 2 * (∑ i, a i ^ 2) ^ 3 := by sorry
