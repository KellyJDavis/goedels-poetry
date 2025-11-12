
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2012 A7

Let $R$ be a totally ordered commutative ring and $\sigma$ be a set of variables.
Let $R[\sigma]$ denote the set of multivariate polynomials in these variables. A function
$f : R^\sigma \to R$ is called a **metapolynomial** if it can be represented as
$$ f(\mathbf{x}) = \max_{i \le m} \min_{j \le n_i} P_{i, j}(\mathbf{x}) $$
for some polynomials $P_{i, j} \in R[\sigma]$. This set of functions is the "meta-closure"
of the set of functions represented by polynomials.

Prove that the set of metapolynomials forms a subring of the ring of all functions
from $R^\sigma$ to $R$.
-/
inductive BinOpClosure {α : Type*} (op : α → α → α) (P : α → Prop) : α → Prop where
  | ofMem {a} (h : P a) : BinOpClosure op P a
  | ofOp {a b} (ha : BinOpClosure op P a) (hb : BinOpClosure op P b) : BinOpClosure op P (op a b)


def MetaClosure {α : Type*} [Lattice α] (S : Set α) : Set α :=
  {x | BinOpClosure (· ⊔ ·) (BinOpClosure (· ⊓ ·) (· ∈ S)) x}

abbrev MvPolynomialImage (σ R : Type*) [CommRing R] : Subring ((σ → R) → R) :=
  (Pi.ringHom (MvPolynomial.eval (R := R) (σ := σ))).range

theorem imo_sl_2012_A7 (σ R : Type*) [LinearOrderedCommRing R] :
  ∃ (T : Subring ((σ → R) → R)), T.carrier = MetaClosure (MvPolynomialImage σ R).carrier := by sorry
