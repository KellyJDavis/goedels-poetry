
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-
# IMO 2014 N2

Determine all pairs $(x, y)$ of integers such that
$$ 7x^2 - 13xy + y^2 = (|x - y| + 1)^3. $$
-/
def good (x y : ℤ) := 7 * x ^ 2 - 13 * x * y + 7 * y ^ 2 = (|x - y| + 1) ^ 3

theorem imo_sl_2014_N2 :
    good x y ↔ (∃ m, (x, y) = (m ^ 3 + 2 * m ^ 2 - m - 1, m ^ 3 + m ^ 2 - 2 * m - 1)) ∨
      (∃ m, (x, y) = (m ^ 3 + m ^ 2 - 2 * m - 1, m ^ 3 + 2 * m ^ 2 - m - 1)) := by sorry
