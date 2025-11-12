
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2015 A2

Find all functions $f : ℤ → ℤ$ such that, for any $x, y ∈ ℤ$,
$$ f(x - f(y)) = f(f(x)) - f(y) - 1. $$
-/
/- special open -/ open Finset
theorem imo_sl_2015_A2 {f : Int → Int} :
    (∀ x y, f (x - f y) = f (f x) - f y - 1) ↔ (f = λ _ ↦ -1) ∨ f = (· + 1) := by sorry
