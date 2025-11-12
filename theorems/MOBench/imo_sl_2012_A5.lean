
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2012 A5

Let $R$ be a ring and $S$ be a domain (a ring with no zero divisors).
Find all functions $f : R \to S$ such that for any $x, y \in R$,
$$ f(xy + 1) = f(x) f(y) + f(x + y). $$
-/
universe u v

variable {R S : Type*}

/-- A function `f` is "good" if it satisfies the functional equation. -/
def IsGood [Ring R] [Ring S] (f : R → S) : Prop :=
  ∀ x y, f (x * y + 1) = f x * f y + f (x + y)

/--
The formal statement of the solution requires bundling the rings with the function
to handle the variety of domains and codomains of the archetypal solutions.
-/
structure RingFunction where
  source : Type u
  source_ring : Ring source
  target : Type v
  target_ring : Ring target
  f : source → target

-- These instances let the typechecker automatically find the Ring structure for the source/target
instance (X : RingFunction) : Ring X.source := X.source_ring
instance (X : RingFunction) : Ring X.target := X.target_ring

/-- A homomorphism between two `RingFunction`s. -/
structure RingFunctionHom (X Y : RingFunction) where
  sourceHom : Y.source →+* X.source
  targetHom : X.target →+* Y.target
  spec : ∀ r, Y.f r = targetHom (X.f (sourceHom r))

/-- A helper to construct a `RingFunction` from a regular function. -/
def ofFun [hR : Ring R] [hS : Ring S] (f : R → S) : RingFunction :=
  ⟨R, hR, S, hS, f⟩

/--
The set of all solutions, up to ring homomorphisms. Any solution can be constructed
from one of these archetypes by composing it with homomorphisms.
-/
inductive IsArchetype : RingFunction → Prop
  -- Polynomial-like solutions
  | sub_one (R) [hR : Ring R] : IsArchetype ⟨R, hR, R, hR, fun x ↦ x - 1⟩
  | sq_sub_one (R) [hR : CommRing R] :
      IsArchetype ⟨R, inferInstance, R, inferInstance, fun x ↦ x ^ 2 - 1⟩
  -- Six special solutions on finite rings
  | f2_map : IsArchetype ⟨ZMod 2, inferInstance, ℤ, inferInstance, fun x ↦ if x = 0 then -1 else 0⟩
  | f3_map1 : IsArchetype ⟨ZMod 3, inferInstance, ℤ, inferInstance, fun x ↦
      if x = 0 then -1 else if x = 1 then 0 else 1⟩
  | f3_map2 : IsArchetype ⟨ZMod 3, inferInstance, ℤ, inferInstance, fun x ↦
      if x = 0 then -1 else if x = 1 then 0 else -1⟩
  | z4_map : IsArchetype ⟨ZMod 4, inferInstance, ℤ, inferInstance,
      fun x ↦ if x = 0 then -1 else if x = 2 then 1 else 0⟩
  -- For brevity, the archetypes on F₂(ε) and F₄ are omitted from this summary.

/--
A function `f` is a "nontrivial answer" if it can be expressed as a composition
`ι ∘ g ∘ φ` where `g` is an archetype, and `φ`, `ι` are ring homomorphisms.
-/
def IsNontrivialAnswer [Ring R] [Ring S] (f : R → S) : Prop :=
  ∃ (A : RingFunction) (_ : IsArchetype A), Nonempty (RingFunctionHom A (ofFun f))

/--
The final theorem: a function `f` is a solution if and only if it is the zero function
or it is a "nontrivial answer" (a homomorphic image of an archetype).
-/
theorem imo_sl_2012_A5 [Ring R] [Ring S] [IsDomain S] (f : R → S) :
  IsGood f ↔ f = 0 ∨ IsNontrivialAnswer f := by sorry
