
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2023 A2

Let $G$ be a $2$-divisible abelian group and $R$ be a totally ordered ring.
Let $f : G → R$ be a function such that
* $f(x + y) f(x - y) ≥ f(x)^2 - f(y)^2$ for all $x, y ∈ G$,
* $f(x_0 + y_0) f(x_0 - y_0) > f(x_0)^2 - f(y_0)^2$ for some $x_0, y_0 ∈ G$.

Prove that either $f ≥ 0$ or $f ≤ 0$.
-/
theorem imo_sl_2023_A2 [AddCommGroup G] (hG : ∀ x : G, ∃ y, 2 • y = x) [LinearOrderedRing R]
    {f : G → R} (hf : ∀ x y, f x ^ 2 - f y ^ 2 ≤ f (x + y) * f (x - y))
    (hf0 : ∃ x0 y0, f x0 ^ 2 - f y0 ^ 2 < f (x0 + y0) * f (x0 - y0)) :
    (∀ x, 0 ≤ f x) ∨ (∀ x, f x ≤ 0) := by sorry
