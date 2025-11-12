
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2006 N1 (P4)

Determine all pairs $(x, y) ∈ ℕ × ℤ$ such that $1 + 2^x + 2^{2x + 1} = y^2$.
-/
/- special open -/ open Finset
def good (x : ℕ) (y : ℤ) := 2 ^ (2 * x + 1) + 2 ^ x + 1 = y ^ 2

theorem imo_sl_2006_N1 :
    good x y ↔ (x = 0 ∧ (y = 2 ∨ y = -2)) ∨ (x = 4 ∧ (y = 23 ∨ y = -23)) := by sorry
