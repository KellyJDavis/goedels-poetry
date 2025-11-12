
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2009 A3

Find all functions $f : \mathbb{N} \to \mathbb{N}$ such that for any $x, y \in \mathbb{N}$, the numbers
$x$, $f(y)$, and $f(y + f(x))$ form the sides of a possibly degenerate triangle.

-/
structure IsNatTriangle (x y z : ℕ) : Prop where
  side_x : x ≤ y + z
  side_y : y ≤ z + x
  side_z : z ≤ x + y

def IsGoodNat (f : ℕ → ℕ) : Prop :=
  ∀ x y, IsNatTriangle x (f y) (f (y + f x))

theorem imo_sl_2009_A3a_nat (f : ℕ → ℕ) : IsGoodNat f ↔ f = id := by sorry
