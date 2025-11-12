
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2009 A5

Let $R$ be a totally ordered ring.
Prove that there does not exist a function $f : R → R$ such that for all $x, y ∈ R$,
$$ f(x - f(y)) ≤ y f(x) + x. $$
-/
theorem imo_sl_2009_A5 [LinearOrderedRing R] (f : R → R) :
    ¬∀ x y, f (x - f y) ≤ y * f x + x := by sorry
