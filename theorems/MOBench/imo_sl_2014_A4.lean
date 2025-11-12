
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2014 A4

Let $b$ and $c$ be integers with $|b| > 1$ and $c ≠ 0$.
Find all functions $f : ℤ → ℤ$ such that, for any $x, y ∈ ℤ$,
$$ f(y + f(x)) - f(y) = f(bx) - f(x) + c. $$
-/
/- special open -/ open Finset
def good (b c : ℤ) (f : ℤ → ℤ) := ∀ x y : ℤ, f (y + f x) - f y = f (b * x) - f x + c

theorem imo_sl_2014_A4 {b c : ℤ} (h : 1 < b.natAbs) (h0 : c ≠ 0) :
    good b c f ↔ b - 1 ∣ c ∧ f = ((b - 1) * · + c / (b - 1)) := by sorry
