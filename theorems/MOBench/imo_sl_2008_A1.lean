
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2008 A1 (P4)

Let $R$ be a totally ordered commutative ring, and let $R_{>0} = \{x ∈ R : x > 0\}$.
Find all functions $f : R_{>0} → R_{>0}$ such that for any $p, q, r, s > 0$ with $pq = rs$,
$$ (f(p)^2 + f(q)^2) (r^2 + s^2) = (p^2 + q^2) (f(r^2) + f(s^2)). $$
-/
/- special open -/ open Finset
structure weakGood [OrderedSemiring R] (f : R → R) : Prop where
  map_pos_of_pos : ∀ x > 0, f x > 0
  good' : ∀ p > 0, ∀ q > 0, ∀ r > 0, ∀ s > 0, p * q = r * s →
    (f p ^ 2 + f q ^ 2) * (r ^ 2 + s ^ 2) = (p ^ 2 + q ^ 2) * (f (r ^ 2) + f (s ^ 2))

variable [LinearOrderedField R]

def good (f : {x : R // 0 < x} → {x : R // 0 < x}) :=
  ∀ p q r s, p * q = r * s →
    (f p ^ 2 + f q ^ 2) * (r ^ 2 + s ^ 2) = (p ^ 2 + q ^ 2) * (f (r ^ 2) + f (s ^ 2))

theorem imo_sl_2008_A1 [ExistsAddOfLE R] {f : {x : R // 0 < x} → {x : R // 0 < x}} :
    good f ↔ f = id ∨ ∀ x, x * f x = 1 := by sorry
