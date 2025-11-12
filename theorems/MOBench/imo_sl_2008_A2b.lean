
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2008 A2

2. Show that there exists infinitely many triplets $(x, y, z) \in (\mathbb{Q} \setminus \{1\})^3$
  with $xyz = 1$ such that the above inequality becomes equality.
-/
structure IsGood (p : Fin 3 → ℚ) : Prop where
  p_ne_one : ∀ i, p i ≠ 1
  p_mul_eq_one : p 0 * p 1 * p 2 = 1
  spec : (p 0 / (p 0 - 1)) ^ 2 + (p 1 / (p 1 - 1)) ^ 2 + (p 2 / (p 2 - 1)) ^ 2 = 1

theorem imo_sl_2008_A2b_part2 : {p : Fin 3 → ℚ | IsGood p}.Infinite := by sorry
