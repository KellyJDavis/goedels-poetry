
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2015 N2

Let $a, b ∈ ℕ$ such that $a! + b! ∣ a! b!$.
Prove that $3a ≥ 2b + 2$, and find all the equality cases.
-/
/- special open -/ open Finset
def good (c d : ℕ) := c + d ∣ c * d

theorem imo_sl_2015_N2 (h : good a.factorial b.factorial) :
    2 * b + 2 ≤ 3 * a ∧ (2 * b + 2 = 3 * a ↔ a = 2 ∧ b = 2 ∨ a = 4 ∧ b = 5) := by sorry
