import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- What is the largest negative integer $x$ satisfying $$24x \equiv 15 \pmod{1199}~?$$ Show that it is -449.-/
theorem mathd_numbertheory_765 (x : ℤ) (h₀ : x < 0) (h₁ : 24 * x % 1199 = 15) : x ≤ -449 := by sorry
