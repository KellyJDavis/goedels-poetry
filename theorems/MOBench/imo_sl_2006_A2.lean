
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2006 A2

Consider the sequence $(a_n)_{n ≥ 0}$ of rational nuimbers defined by $a_0 = 1$ and
$$ a_n = -\sum_{k = 0}^{n - 1} \frac{a_k}{n + 1 - k}. $$
Prove that $a_n > 0$ for all $n ≠ 0$.
-/
/- special open -/ open Finset
def a : ℕ → ℚ
  | 0 => -1
  | n + 1 => -(univ : Finset (Fin (n + 1))).sum λ i ↦ a i / (n + 2 - i : ℕ)

theorem imo_sl_2006_A2 (h : n ≠ 0) : 0 < a n := by sorry
