import Mathlib.Algebra.BigOperators.Intervals
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Tactic

/-!
# Hungarian Mathematical Olympiad 1998, Problem 6

Let x, y, z be integers with z > 1. Show that

 (x + 1)² + (x + 2)² + ... + (x + 99)² ≠ yᶻ.
-/

namespace Hungary1998P6

theorem hungary1998_p6 (x y : ℤ) (z : ℕ) (hz : 1 < z) :
    ∑ i ∈ Finset.range 99, (x + i + 1)^2 ≠ y^z := sorry


end Hungary1998P6
