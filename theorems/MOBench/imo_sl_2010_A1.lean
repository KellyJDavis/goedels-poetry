
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2010 A1

Let $R$ and $S$ be totally ordered rings with a floor function (i.e., `FloorRing`s).
Find all functions $f : R → S$ such that for any $x, y \in R$,
$$ f(\lfloor x \rfloor y) = f(x) \lfloor f(y) \rfloor. $$
-/
/- special open -/ open Classical
/-- A function `f` is "good" if it satisfies the functional equation. -/
def IsGood [LinearOrderedRing R] [FloorRing R] [LinearOrderedRing S] [FloorRing S] (f : R → S) : Prop :=
  ∀ x y, f (⌈x⌉ • y) = f x * ⌈f y⌉

/--
A helper definition for the discrete case: `ε` is "infinitesimal" if all its
natural number multiples are less than 1 in absolute value.
-/
def IsInfinitesimal [LinearOrderedRing S] (ε : S) : Prop :=
  ∀ n : ℕ, n • |ε| < 1

/--
For the case where `R` is isomorphic to `ℤ`, the solutions fall into one of
three families, captured by this inductive proposition.
-/
inductive IsAnswer [LinearOrderedRing R] [MulOneClass R]
    [LinearOrderedRing S] [FloorRing S] : (R → S) → Prop
  /-- Solutions that are integer-valued monoid homomorphisms. -/
  | MonoidHom_cast (phi : R →* ℤ) :
      IsAnswer (fun x ↦ (phi x : S))
  /-- Solutions of the form `n ↦ (1 + ε)^n`, where `ε` is a positive infinitesimal. -/
  | one_add_ε (ε : S) (_ : 0 < ε) (_ : IsInfinitesimal ε) (phi : R →* ℕ) :
      IsAnswer (fun x ↦ phi x • (1 + ε))
  /-- Solutions that are indicator functions on submonoids of `R`. -/
  | indicator (A : Set R) (_ : ∀ m n, m * n ∈ A ↔ m ∈ A ∧ n ∈ A) (C : S) (_ : ⌈C⌉ = 1) :
      IsAnswer (fun x ↦ if x ∈ A then C else 0)

/--
The final solution, which splits depending on the properties of the domain `R`.
-/
theorem imo_sl_2010_A1 [LinearOrderedRing R] [FloorRing R]
    [LinearOrderedRing S] [FloorRing S] (f : R → S) :
  IsGood f ↔ if DenselyOrdered R then (∃ C, ⌈C⌉ = 1 ∧ f = fun _ ↦ C) ∨ f = (fun _ ↦ 0)
    else IsAnswer f := by sorry
