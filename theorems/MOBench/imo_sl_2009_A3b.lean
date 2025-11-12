
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2009 A3

The original problem statement. Find all functions $f : \mathbb{N}^+ \to \mathbb{N}^+$ such that for any
$x, y \in \mathbb{N}^+$, the numbers $x$, $f(y)$, and $f(y + f(x) - 1)$ form the sides of a
non-degenerate triangle.
-/
structure IsPNatTriangle (x y z : ℕ+) : Prop where
  side_x : x < y + z
  side_y : y < z + x
  side_z : z < x + y

def IsGoodPNat (f : ℕ+ → ℕ+) : Prop :=
  ∀ x y, IsPNatTriangle x (f y) (f (y + f x - 1))

theorem imo_sl_2009_A3b_pnat (f : ℕ+ → ℕ+) : IsGoodPNat f ↔ f = id := by sorry
