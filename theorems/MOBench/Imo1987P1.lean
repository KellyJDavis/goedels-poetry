
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1987, Problem 1

Let $p_{n, k}$ be the number of permutations of a set of cardinality `n ≥ 1`
that fix exactly `k` elements. Prove that $∑_{k=0}^n k p_{n,k}=n!$.
-/
/-- Given `α : Type*` and `k : ℕ`, `fiber α k` is the set of permutations of
    `α` with exactly `k` fixed points. -/
def fiber (α : Type*) [Fintype α] [DecidableEq α] (k : ℕ) : Set (Equiv.Perm α) :=
  {σ : Equiv.Perm α | Fintype.card (Function.fixedPoints σ) = k}

instance {k : ℕ} (α : Type*) [Fintype α] [DecidableEq α] :
  Fintype (fiber α k) := by unfold fiber; infer_instance

/-- `p α k` is the number of permutations of `α` with exactly `k` fixed points. -/
def p (α : Type*) [Fintype α] [DecidableEq α] (k : ℕ) : ℕ := Fintype.card (fiber α k)


theorem imo1987_p1 {n : ℕ} (hn : 1 ≤ n) :
    ∑ k ∈ Finset.range (n + 1), k * p (Fin n) k = n ! := by sorry
