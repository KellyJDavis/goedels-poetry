import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Fintype.Perm
import Mathlib.Data.Fintype.Prod
import Mathlib.Dynamics.FixedPoints.Basic

/-!
# International Mathematical Olympiad 1987, Problem 1

Let $p_{n, k}$ be the number of permutations of a set of cardinality `n ≥ 1` that fix exactly `k`
elements. Prove that $∑_{k=0}^n k p_{n,k}=n!$.
-/

namespace Imo1987P1

open scoped Nat
open Finset (range)


theorem imo1987_p1 {n : ℕ} (hn : 1 ≤ n) : ∑ k ∈ range (n + 1), k * p (Fin n) k = n ! := sorry



end Imo1987P1
