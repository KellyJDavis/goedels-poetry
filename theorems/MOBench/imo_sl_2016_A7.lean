
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2016 A7

Let $R$ be a ring and $S$ be a totally ordered commutative ring.
Find all functions $f : R \to S$ such that for any $x, y \in R$,
$$ f(x + y)^2 = 2 f(x) f(y) + \max\{f(x^2) + f(y^2), f(x^2 + y^2)\}. $$
-/
variable {R : Type*} [Ring R] {S : Type*} [LinearOrderedCommRing S]

/-- A function `f` is a solution if it satisfies the functional equation. -/
def IsSolution (f : R → S) : Prop :=
  ∀ x y : R, f (x + y) ^ 2 = 2 * f x * f y + max (f (x ^ 2) + f (y ^ 2)) (f (x ^ 2 + y ^ 2))

theorem imo_sl_2016_A7 (f : R → S) :
  IsSolution f ↔
    (f = (fun _ ↦ (0 : S)) ∨ ∃ (phi : RingHom R S), f = phi) ∨
    (f = (fun _ ↦ (-1 : S)) ∨ ∃ (phi : RingHom R S), f = (fun x ↦ phi x - 1)) := by sorry
