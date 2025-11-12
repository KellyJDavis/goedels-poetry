
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2017 A8

Let $G$ be a totally ordered abelian group. We are interested in functions $f : G \to G$
that satisfy the following property: for any $x, y \in G$,
$$ \text{if } f(x) + y < f(y) + x, \text{ then } f(x) + y \le 0 \le f(y) + x. $$

The problem is to show that if $G$ is densely ordered, then every function $f$ with the
above property also satisfies:
$$ f(y) + x \le f(x) + y \quad \text{for all } x, y \in G \text{ with } x \le y. $$

-/
def IsGood {G : Type*} [AddCommGroup G] [LinearOrder G] (f : G → G) : Prop :=
  ∀ x y : G, f x + y < f y + x → f x + y ≤ 0 ∧ 0 ≤ f y + x

theorem imo_sl_2017_A8 {G : Type*} [AddCommGroup G] [LinearOrder G] :
  (∀ (f : G → G), IsGood f → (∀ x y, x ≤ y → f y + x ≤ f x + y)) ↔ DenselyOrdered G := by sorry
