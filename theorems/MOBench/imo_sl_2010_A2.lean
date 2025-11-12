
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2010 A2

Let $R$ be a totally ordered commutative ring.
Fix some $a, b, c, d \in R$ such that $a + b + c + d = 6$ and $a^2 + b^2 + c^2 + d^2 = 12$.
Prove that
$$ 36 \le 4(a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) \le 48. $$
-/
theorem imo_sl_2010_A2 [LinearOrderedCommRing R] (a b c d : R)
    (h_sum : a + b + c + d = 6)
    (h_sq_sum : a ^ 2 + b ^ 2 + c ^ 2 + d ^ 2 = 12) :
    let S := 4 * (a ^ 3 + b ^ 3 + c ^ 3 + d ^ 3) - (a ^ 4 + b ^ 4 + c ^ 4 + d ^ 4)
    36 ≤ S ∧ S ≤ 48 := by sorry
