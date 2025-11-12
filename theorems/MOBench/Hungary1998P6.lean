
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# Hungarian Mathematical Olympiad 1998, Problem 6

Let x, y, z be integers with z > 1. Show that

 (x + 1)² + (x + 2)² + ... + (x + 99)² ≠ yᶻ.
-/
theorem hungary1998_p6 (x y : ℤ) (z : ℕ) (hz : 1 < z) :
    ∑ i ∈ Finset.range 99, (x + i + 1)^2 ≠ y^z := by sorry
