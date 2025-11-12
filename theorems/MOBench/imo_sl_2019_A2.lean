
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2019 A2

Let $R$ be a totally ordered ring and $x_1, x_2, …, x_n ∈ R$ be elements with
$$ x_1 + x_2 + … + x_n = 0. $$
Let $a, b ∈ R$ such that $b ≤ x_i ≤ a$ for all $i ≤ n$.
Show that $$ nab + \sum_{i = 1}^n x_i^2 ≤ 0. $$
-/
/- special open -/ open Multiset
theorem imo_sl_2019_A2 [LinearOrderedCommRing R]
    {a b : R} {M : Multiset R} (hM : M.sum = 0) (ha : ∀ x ∈ M, x ≤ a) (hb : ∀ x ∈ M, b ≤ x) :
    card M • (a * b) + (M.map λ x ↦ x ^ 2).sum ≤ 0 := by sorry
