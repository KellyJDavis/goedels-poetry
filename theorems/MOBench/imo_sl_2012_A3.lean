
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2012 A3

Let $m \ge 2$ be an integer, $R$ be a totally ordered commutative ring, and
$x_0, x_1, \dots, x_{m-1} \in R$ be positive elements such that
$x_0 x_1 \cdots x_{m-1} = 1$. Prove that
$$ (1 + x_0)^2 (1 + x_1)^3 \cdots (1 + x_{m-1})^{m+1} > (m + 1)^{m+1}. $$
-/
/- special open -/ open Finset
theorem imo_sl_2012_A3 [LinearOrderedCommRing R] (m : Nat) (hm : 2 ≤ m)
    (x : Fin m → R) (hx_pos : ∀ i, 0 < x i) (hx_prod : ∏ i, x i = 1) :
    (m + 1) ^ (m + 1) < ∏ i, (1 + x i) ^ ((i : Nat) + 2) := by sorry
