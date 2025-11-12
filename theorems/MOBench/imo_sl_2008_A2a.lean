
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2008 A2

1. Let $F$ be an ordered field, and consider $x, y, z \in F \setminus \{1\}$ with $xyz = 1$.
Prove that $$ \frac{x^2}{(x - 1)^2} + \frac{y^2}{(y - 1)^2} + \frac{z^2}{(z - 1)^2} \ge 1. $$

-/
theorem imo_sl_2008_A2a_part1 [LinearOrderedField F]
    {x y z : F} (hx : x ≠ 1) (hy : y ≠ 1) (hz : z ≠ 1) (h : x * y * z = 1) :
    1 ≤ (x / (x - 1)) ^ 2 + (y / (y - 1)) ^ 2 + (z / (z - 1)) ^ 2 := by sorry
