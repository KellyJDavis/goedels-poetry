
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2022 A3 (P2)

Let $R$ be a totally ordered commutative ring, and let $R_{>0} = \{x ∈ R : x > 0\}$.
Find all functions $f : R_{>0} → R_{>0}$ such that for any $x ∈ R_{>0}$,
  there exists a unique $y ∈ R_{>0}$ such that $x f(y) + y f(x) ≤ 2$.
-/
variable [LinearOrderedField R]

def good (f : {x : R // 0 < x} → {x : R // 0 < x}) :=
  ∀ x, ∃! y, x * f y + y * f x ≤ ⟨2, two_pos⟩

theorem imo_sl_2022_A3 [ExistsAddOfLE R] {f : {x : R // 0 < x} → {x : R // 0 < x}} :
    good f ↔ ∀ x, x * f x = 1 := by sorry
