
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2015 N3

Let $m$ and $n > 1$ be positive integers such that $k ∣ m$ whenever $n ≤ k < 2n$.
Prove that $L - 1$ is not a power of $2$, where
$$ L = \prod_{k = n}^{2n - 1} \left(\frac{m}{k} + 1\right). $$
-/
/- special open -/ open Finset
theorem imo_sl_2015_N3 (hm : 0 < m) (hn : 1 < n) (h : ∀ k ∈ Ico n (2 * n), k ∣ m) :
    ∀ N, ∏ k ∈ Ico n (2 * n), (m / k + 1) ≠ 2 ^ N + 1 := by sorry
