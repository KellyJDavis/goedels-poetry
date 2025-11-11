import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Show that for positive integer $n$, the remainder of $3^{2^n} - 1 \equiv 2^{n+2} \mod 2^{n+3}$.-/
theorem numbertheory_3pow2pownm1mod2pownp3eq2pownp2 (n : ℕ) (h₀ : 0 < n) :
    (3 ^ 2 ^ n - 1) % 2 ^ (n + 3) = 2 ^ (n + 2) := by sorry
