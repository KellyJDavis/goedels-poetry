
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2017 A1

Let $a_1, a_2, \dots, a_n$ and $k$ be positive integers such that the sum of the reciprocals
of the $a_i$ is equal to $k$, i.e., $\sum_{i=1}^n \frac{1}{a_i} = k$.
Suppose that at least one of the $a_i$ is greater than $1$.

Prove that for any totally ordered commutative semiring $R$ and any positive element $x \in R$,
$$ \left(\prod_{i=1}^n a_i\right) (x + 1)^k < \prod_{i=1}^n (x + a_i). $$
-/
/- special open -/ open Finset
theorem imo_sl_2017_A1 [LinearOrderedField R] (n : ℕ) (a : Fin n → ℕ) (k : ℕ) (x : R)
    (ha_pos : ∀ i, 0 < a i)
    (ha_gt_one : ∃ i, 1 < a i)
    (hk_sum : (∑ i, (a i : ℚ)⁻¹) = k)
    (hx : 0 < x) :
    ((∏ i, a i) : R) * (x + 1) ^ k < ∏ i, (x + (a i : R)) := by sorry
