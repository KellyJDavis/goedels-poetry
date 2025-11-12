
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2007 N1

Find all pairs $(k, n) \in \mathbb{N}^2$ such that $7^k - 3^n \mid k^4 + n^2$.
-/
def good (k n : ℕ) : Prop := (7 : ℤ) ^ k - 3 ^ n ∣ (k ^ 4 + n ^ 2 : ℕ)

theorem imo_sl_2007_N1 (k n : ℕ) :
  good k n ↔ (k = 0 ∧ n = 0) ∨ (k = 2 ∧ n = 4) := by sorry
