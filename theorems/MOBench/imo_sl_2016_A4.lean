
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2016 A4

Let $M$ be an integral multiplicative monoid with a cancellative, distributive addition.
Find all functions $f : M → M$ such that, for all $x, y ∈ M$,
$$ x f(x^2) f(f(y)) + f(y f(x)) = f(xy) \left(f(f(y^2)) + f(f(x^2))\right). $$
-/
def good [Mul M] [Add M] (f : M → M) :=
  ∀ x y, x * f (x * x) * f (f y) + f (y * f x) = f (x * y) * (f (f (y * y)) + f (f (x * x)))

class CancelCommDistribMonoid (M) extends CancelCommMonoid M, Distrib M

variable [CancelCommDistribMonoid M]

theorem imo_sl_2016_A4 [IsCancelAdd M] {f : M → M} : good f ↔ ∀ x, x * f x = 1 := by sorry
