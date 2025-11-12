
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2006 N5

Let $p > 3$ be a prime.
Determine all pairs $(x, y)$ of integers such that
$$ \sum_{k = 0}^{p - 1} x^k = y^{p - 2} - 1. $$
-/
/- special open -/ open Finset
theorem imo_sl_2006_N5 {p : ℕ} (hp : p.Prime) (h : 3 < p) (x y : ℤ) :
    ¬(range p).sum (x ^ ·) = y ^ (p - 2) - 1 := by sorry
