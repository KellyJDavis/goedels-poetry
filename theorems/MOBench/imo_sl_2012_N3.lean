
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2012 N3

Determine all integers $m > 1$ such that $n ∣ \binom{n}{m - 2n}$ for every $n ≤ m/2$.
-/
theorem imo_sl_2012_N3 (h : 1 < m) :
    (∀ n : ℕ, 2 * n ≤ m → n ∣ n.choose (m - 2 * n)) ↔ m.Prime := by sorry
