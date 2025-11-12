
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2009 A2

Let $F$ be a totally ordered field, and let $a, b, c \in F$ be positive elements.
Prove that
$$ \frac{1}{(2a + b + c)^2} + \frac{1}{(2b + c + a)^2} + \frac{1}{(2c + a + b)^2}
  \le \frac{3}{16(ab+bc+ca)}. $$
-/
theorem imo_sl_2009_A2 [LinearOrderedField F]
    {a b c : F} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h_norm : a⁻¹ + b⁻¹ + c⁻¹ = a + b + c) :
    ((2 * a + b + c) ^ 2)⁻¹ + ((2 * b + c + a) ^ 2)⁻¹ + ((2 * c + a + b) ^ 2)⁻¹ ≤ 3 / 16 := by sorry
