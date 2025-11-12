
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2006 N7

Prove that for any $b ∈ ℕ$ and $n ∈ ℕ^+$, there exists $m ∈ ℕ$ such that $n ∣ b^m + m$.
-/
theorem imo_sl_2006_N7 (hn : 0 < n) (b) : ∃ m, n ∣ b ^ m + m := by sorry
