
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2007 N2

Fix integers $b > 0$ and $n ≥ 0$.
Suppose that for each $k ∈ ℕ^+$, there exists an integer $a$ such that $k ∣ b - a^n$.
Prove that $b = A^n$ for some integer $A$.
-/
/- special open -/ open Finset
theorem imo_sl_2007_N2 (h : 0 < b) (h0 : ∀ k : ℕ, 0 < k → ∃ c : ℤ, (k : ℤ) ∣ b - c ^ n) :
    ∃ a : ℤ, b = a ^ n := by sorry
