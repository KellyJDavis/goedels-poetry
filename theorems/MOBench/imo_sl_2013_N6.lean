
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2013 N6

Determine all functions $f : ℚ → ℤ$ such that for any $x ∈ ℚ$, $a ∈ ℤ$, and $b ∈ ℕ^+$,
$$ f\left(\frac{f(x) + a}{b}\right) = f\left(\frac{x + a}{b}\right). $$
-/
def good (f : ℚ → ℤ) :=
    ∀ (x : ℚ) (a : ℤ) (b : ℕ), 0 < b → f ((f x + a) / b) = f ((x + a) / b)


theorem imo_sl_2013_N6 : good f ↔ (∃ C, f = λ _ ↦ C) ∨ f = Int.floor ∨ f = Int.ceil := by sorry
