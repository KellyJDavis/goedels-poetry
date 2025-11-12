
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2011 A6 (P3)

Let $R$ be a totally ordered commutative ring.
Let $f : R → R$ be a function such that, for any $x, y ∈ R$,
$$ f(x + y) ≤ y f(x) + f(f(x)). $$
Show that $f(x) = 0$ for any $x ∈ R$ such that $x ≤ 0$.
-/
theorem imo_sl_2011_A6 [LinearOrderedCommRing R]
    {f : R → R} (h : ∀ x y : R, f (x + y) ≤ y * f x + f (f x)) :
    ∀ x : R, x ≤ 0 → f x = 0 := by sorry
