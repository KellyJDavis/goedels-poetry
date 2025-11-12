
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2012 N1

Let $R$ be a commutative ring.
A set $A ⊆ R$ is called *admissible* if $x^2 + rxy + y^2 ∈ A$ for any $x, y ∈ A$ and $r ∈ R$.
Determine all pairs $(x, y) ∈ R^2$ such that the only
  admissible set containing $x$ and $y$ is $R$.
-/
def admissible [Semiring R] (A : Set R) :=
  ∀ x y : R, x ∈ A → y ∈ A → ∀ r : R, x ^ 2 + r * x * y + y ^ 2 ∈ A

theorem imo_sl_2012_N1 [CommRing R] (x y : R) :
    (∀ A : Set R, admissible A → x ∈ A → y ∈ A → ∀ z : R, z ∈ A) ↔ IsCoprime x y := by sorry
