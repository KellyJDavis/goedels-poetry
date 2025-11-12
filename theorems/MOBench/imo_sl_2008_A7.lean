
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2008 A7

Let $F$ be a totally ordered field.
1. Prove that, for any $a, b, c, d \in F$ positive,
$$ \frac{(a - b)(a - c)}{a + b + c} + \frac{(b - c)(b - d)}{b + c + d} +
  \frac{(c - d)(c - a)}{c + d + a} + \frac{(d - a)(d - b)}{d + a + b} \ge 0. $$
2. Find all cases of equality.
-/
theorem imo_sl_2008_A7 [LinearOrderedField F]
    {a b c d : F} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
    (0 ≤ (a - b) * (a - c) / (a + b + c) + (b - c) * (b - d) / (b + c + d) +
      (c - d) * (c - a) / (c + d + a) + (d - a) * (d - b) / (d + a + b)) ∧
    ((a - b) * (a - c) / (a + b + c) + (b - c) * (b - d) / (b + c + d) +
      (c - d) * (c - a) / (c + d + a) + (d - a) * (d - b) / (d + a + b) = 0 ↔
      a = c ∧ b = d) := by sorry
