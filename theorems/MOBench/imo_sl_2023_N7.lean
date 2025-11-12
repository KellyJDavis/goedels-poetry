
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2023 N7

Find all possible values of $a + b + c + d$ across all $a, b, c, d ∈ ℕ^+$ satisfying
$$ \frac{ab}{a + b} + \frac{cd}{c + d} = \frac{(a + b)(c + d)}{a + b + c + d}. $$
-/
class nice (a b c d : ℕ) : Prop where
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  d_pos : 0 < d
  spec : ((a * b : ℕ) : ℚ) / (a + b : ℕ) + (c * d : ℕ) / (c + d : ℕ)
    = (a + b : ℕ) * (c + d : ℕ) / (a + b + c + d : ℕ)

theorem imo_sl_2023_N7 (hn : 0 < n) :
    (∃ a b c d, nice a b c d ∧ a + b + c + d = n) ↔ ¬Squarefree n := by sorry
