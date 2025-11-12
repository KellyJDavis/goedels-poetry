
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2022 N8

Given $n ∈ ℕ$ such that $2^n + 65 ∣ 5^n - 3^n$, prove that $n = 0$.
-/
theorem imo_sl_2022_N8 (h : 5 ^ n ≡ 3 ^ n [MOD 2 ^ n + 65]) : n = 0 := by sorry
