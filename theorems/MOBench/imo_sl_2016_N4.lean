
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2016 N4

Consider some $k, ℓ, m, n ∈ ℕ^+$ with $n > 1$ such that
$$ n^k + mn^ℓ + 1 ∣ n^{k + ℓ} - 1. $$
Prove that one of the following holds:
* $m = 1$ and $ℓ = 2k$; or
* $k = (t + 1)ℓ$ and $m(n^ℓ - 1) = n^{t ℓ} - 1$ for some $t > 0$.
-/
theorem imo_sl_2016_N4 (hk : 0 < k) (hl : 0 < l) (hm : 0 < m) (hn : 1 < n)
    (h : n ^ k + m * n ^ l + 1 ∣ n ^ (k + l) - 1) :
    (m = 1 ∧ l = 2 * k) ∨ (∃ t > 0, k = (t + 1) * l ∧ m * (n ^ l - 1) = n ^ (l * t) - 1) := by sorry
