
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2015, Problem 6
The sequence $a_1, a_2, \dots$ of integers satisfies the conditions
1. $1 ≤ a_j ≤ 2015$ for all $j ≥ 1$,
2. $k + a_k ≠ l + a_l$ for all $1 ≤ k < l$.
Prove that there exist two positive integers $b$ and $N$ for which
$$\left|\sum_{j=m+1}^n (a_j-b)\right| ≤ 1007^2$$
for all integers $m,n$ such that $N ≤ m < n$.
-/
/-- The conditions on `a` in the problem. We reindex `a` to start from 0 rather than 1;
`N` then only has to be nonnegative rather than positive, and the sum in the problem statement
is over `Ico m n` rather than `Ioc m n`. -/
def Condition (a : ℕ → ℤ) : Prop :=
  (∀ i, a i ∈ Finset.Icc 1 2015) ∧ Function.Injective fun i ↦ i + a i


theorem imo2015_p6 (ha : Condition a) :
    ∃ b > 0, ∃ N, ∀ m ≥ N, ∀ n > m, |∑ j ∈ Finset.Ico m n, (a j - b)| ≤ 1007 ^ 2 := by sorry
