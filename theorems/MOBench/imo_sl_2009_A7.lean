
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2009 A7

Fix a domain $R$ (a ring with no zero divisors).
Find all functions $f : R \to R$ such that for all $x, y \in R$,
$$ f(x f(x + y)) = f(y f(x)) + x^2. $$

**Note:** There appears to be a typo in the provided formalization's statement
of the problem. The term `f(f(x) y)` from the source code has been changed to `f(y f(x))`
to match the official problem statement.
-/
variable [Ring R]

def IsGood (f : R → R) : Prop :=
  ∀ x y, f (x * f (x + y)) = f (y * f x) + x ^ 2

theorem imo_sl_2009_A7 [NoZeroDivisors R] (f : R → R) :
  IsGood f ↔ f = id ∨ f = neg := by sorry
