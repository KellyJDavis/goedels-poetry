
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2013 N2

Prove that for any positive integers $k, n$, there exist positive integers $m_1, m_2, \dots, m_k$
such that
$$ 1 + \frac{2^k - 1}{n} = \prod_{i = 1}^k \left(1 + \frac{1}{m_i}\right). $$
-/
/- special open -/ open Finset
theorem imo_sl_2013_N2 (k n : ℕ+) :
  ∃ (m : Fin k → ℕ+), (1 : ℚ) + ((2 : ℚ) ^ (k : ℕ) - 1) / (n : ℚ) =
    ∏ i : Fin k, (1 + 1 / (m i : ℚ)) := by sorry
