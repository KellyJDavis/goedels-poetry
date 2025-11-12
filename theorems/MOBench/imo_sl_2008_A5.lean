
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2008 A5

Let $F$ be a totally ordered field and $a_1, a_2, a_3, a_4 \in F$ be positive elements.
Suppose that $a_1 a_2 a_3 a_4 = 1$ and
$$ \sum_{i = 1}^4 \frac{a_i}{a_{i + 1}} < \sum_{i = 1}^4 a_i. $$
Prove that
$$ \sum_{i = 1}^4 a_i < \sum_{i = 1}^4 \frac{a_{i + 1}}{a_i}. $$
-/
theorem imo_sl_2008_A5 [LinearOrderedField F]
    {a b c d : F} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
    (h_prod : a * b * c * d = 1)
    (h_ineq : a / b + b / c + c / d + d / a < a + b + c + d) :
    a + b + c + d < b / a + c / b + d / c + a / d := by sorry
