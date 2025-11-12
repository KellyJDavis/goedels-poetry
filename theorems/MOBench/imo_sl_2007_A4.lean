
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2007 A4

Let $G$ be a totally ordered abelian group and $G_{>0} = \{x \in G : x > 0\}$.
Find all functions $f : G_{>0} \to G_{>0}$ such that for any $x, y \in G_{>0}$,
$$ f(x + f(y)) = f(x + y) + f(y). $$
-/
variable [LinearOrderedAddCommGroup G]

/--
This defines the property of a function `f` satisfying the given functional equation
on the subtype of positive elements `{x : G // 0 < x}`.
-/
def IsGood (f : {x : G // 0 < x} → {x : G // 0 < x}) : Prop :=
  ∀ x y, f (x + f y) = f (x + y) + f y

/--
The solutions to the functional equation are functions that double the input.
-/
theorem imo_sl_2007_A4 (f : {x : G // 0 < x} → {x : G // 0 < x}) :
  IsGood f ↔ f = (fun x ↦ x + x) := by sorry
