
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2011 A3

Let $R$ be a commutative ring where $2$ is not a zero divisor.
Find all functions $f, g : R → R$ such that for any $x, y ∈ R$,
$$ g(f(x + y)) = f(x) + (2x + y) g(y). $$
-/
def good [NonUnitalNonAssocSemiring R] (f g : R → R) :=
  ∀ x y, g (f (x + y)) = f x + (2 • x + y) * g y

theorem imo_sl_2011_A3 [CommRing R] [IsDomain R] (hR : (2 : R) ≠ 0) {f g : R → R} :
    good f g ↔ (f, g) = (λ _ ↦ 0, λ _ ↦ 0) ∨ ∃ c, (f, g) = (λ x ↦ x * x + c, id) := by sorry
