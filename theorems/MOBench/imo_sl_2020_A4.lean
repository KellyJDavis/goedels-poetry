
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2020 A4

Let $a, b, c, d$ be real numbers with $a \ge b \ge c \ge d > 0$ and $a + b + c + d = 1$.
Prove that
$$ (a + 2b + 3c + 4d) a^a b^b c^c d^d < 1. $$
-/
/- special open -/ open NNReal
theorem imo_sl_2020_A4 (a b c d : NNReal)
    (h_ord : a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d > 0)
    (h_sum : a + b + c + d = 1) :
    (a + 2 * b + 3 * c + 4 * d) *
      (a.toReal ^ a.toReal * b.toReal ^ b.toReal * c.toReal ^ c.toReal * d.toReal ^ d.toReal) < 1 := by sorry
