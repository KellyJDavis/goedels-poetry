
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2020 A6

Find all functions $f : ℤ → ℤ$ such that, for any $a, b ∈ ℤ$,
$$ f^{a^2 + b^2}(a + b) = a f(a) + b f(b). $$
-/
/- special open -/ open Function
def good (f : ℤ → ℤ) := ∀ a b, f^[a.natAbs ^ 2 + b.natAbs ^ 2] (a + b) = a * f a + b * f b

theorem imo_sl_2020_A6 {f : ℤ → ℤ} : good f ↔ f = (· + 1) ∨ f = λ _ ↦ 0 := by sorry
