
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2016 A1

Let $R$ be a totally ordered commutative ring.
Let $a_1, a_2, \dots, a_n, c \in R$ be non-negative elements such that $a_i a_j \ge c$
whenever $i
e j$.
Let $r \in R$ be an element such that $n r \ge a_1 + a_2 + \dots + a_n$.
Prove that
$$ \prod_{i = 1}^n (a_i^2 + c) \le (r^2 + c)^n. $$
-/
/- special open -/ open Finset
theorem imo_sl_2016_A1 [LinearOrderedCommRing R] (n : ℕ) (a : Fin n → R) (c r : R)
    (ha : ∀ i, 0 ≤ a i)
    (hc : 0 ≤ c)
    (h_prod_ge : ∀ i j, i ≠ j → c ≤ a i * a j)
    (hr_ge_avg : ∑ i, a i ≤ n • r) :
    ∏ i, (a i ^ 2 + c) ≤ (r ^ 2 + c) ^ n := by sorry
