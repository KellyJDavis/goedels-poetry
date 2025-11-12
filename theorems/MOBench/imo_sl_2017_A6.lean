
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2017 A6

Let $R$ be a ring, $G$ be an abelian (additive) group, and $\iota : G \to R$ be a group
homomorphism. Find all functions $f : R \to G$ such that for any $x, y \in R$,
$$ f(\iota(f(x)) \iota(f(y))) + f(x + y) = f(xy). $$
-/
variable {R G : Type*}

def IsGoodFun [Ring R] [AddCommGroup G] (ι : G →+ R) (f : R → G) : Prop :=
  ∀ x y : R, f (ι (f x) * ι (f y)) + f (x + y) = f (x * y)

@[ext] structure CentralInvolutive (R : Type*) [Ring R] where
  val : R
  val_mul_self_eq_one : val * val = 1
  val_mul_comm (x : R) : x * val = val * x

theorem imo_sl_2017_A6 [Ring R] [AddCommGroup G]
    (hG2 : ∀ x y : G, 2 • x = 2 • y → x = y)
    (hG3 : ∀ x y : G, 3 • x = 3 • y → x = y)
    (ι : G →+ R) (f : R → G) :
    IsGoodFun ι f ↔
      ∃ (rc : RingCon R) (a : CentralInvolutive (rc.Quotient))
        (phi : {ψ : rc.Quotient →+ G // ∀ x, rc.toQuotient (ι (ψ x)) = x}),
        f = fun x ↦ phi.1 (a.val * (rc.toQuotient (1 - x))) := by sorry
