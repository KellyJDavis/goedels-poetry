
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2019 A1 (P1)

Fix an integer $N ≠ 0$.
Find all functions $f : ℤ → ℤ$ such that, for any $a, b ∈ ℤ$,
$$ f(Na) + N f(b) = f(f(a + b)). $$
-/
theorem imo_sl_2019_A1 (h : N ≠ 0) {f : Int → Int} :
    (∀ a b : Int, f (N * a) + N * f b = f (f (a + b)))
      ↔ (f = λ _ ↦ 0) ∨ ∃ c : Int, f = (N * · + c) := by sorry
