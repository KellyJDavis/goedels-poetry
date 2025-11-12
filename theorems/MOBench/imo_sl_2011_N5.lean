
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2011 N5

Let $G$ be an additive group. Find all functions $f : G \to \mathbb{Z}$ such that
for any $x, y \in G$:
1. $f(x) > 0$
2. $f(x - y)$ divides $f(x) - f(y)$.
-/
variable [AddGroup G]

/--
A function `f` is "good" if it satisfies the conditions of the problem.
The codomain is taken to be `ℤ` with a positivity constraint, which is
equivalent to the original problem's `ℕ+` codomain.
-/
structure IsGood (f : G → ℤ) : Prop where
  pos : ∀ x, 0 < f x
  dvd : ∀ x y, f (x - y) ∣ f x - f y

/--
This theorem establishes a key property of any solution `f`.
It shows that if `f(x) ≤ f(y)`, then `f(x)` must divide `f(y)`.
This implies that the set of values taken by `f` must form a divisor chain.
-/
theorem solution_property {f : G → ℤ} (hf : IsGood f) {x y : G} (h_le : f x ≤ f y) :
  f x ∣ f y := by sorry
