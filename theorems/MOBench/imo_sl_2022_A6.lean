
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2022 A6

Let $G$ be a commutative group.
A function $f : G → G$ is called *infectious* if
$$ f(x + f(y)) = f(x) + f(y) \quad ∀ x, y ∈ G. $$
Find all pairs $(m, n)$ of integers such that for any infectious functions
  $f : G → G$, there exists $z ∈ G$ such that $m f(z) = nz$.
-/
structure InfectiousFun (G) [Add G] where
  toFun : G → G
  infectious_def' : ∀ x y, toFun (x + toFun y) = toFun x + toFun y

instance [Add G] : FunLike (InfectiousFun G) G G where
  coe f := f.toFun
  coe_injective' f g h := by rwa [InfectiousFun.mk.injEq]

def good (G) [AddGroup G] (m n : ℤ) := ∀ f : InfectiousFun G, ∃ z, m • f z = n • z

theorem imo_sl_2022_A6 [AddCommGroup G] :
    good G m n ↔ ∀ g : G, (m - n).gcd (addOrderOf g) ∣ m.natAbs := by sorry
