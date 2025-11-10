import Mathlib.Data.Int.Lemmas
import Mathlib.Data.Nat.Fib.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.LinearCombination

open Int Nat Set

namespace Imo1981P3

/-!
# International Mathematical Olympiad 1981, Problem 3

Determine the maximum value of `m ^ 2 + n ^ 2`, where `m` and `n` are integers in
`{1, 2, ..., 1981}` and `(n ^ 2 - m * n - m ^ 2) ^ 2 = 1`.
-/

/-
We generalize the problem to `{1, 2, ..., N}` and then specialize to `N = 1981`.
-/
variable (N : ℕ)

-- N = 1981
@[mk_iff]
structure ProblemPredicate (m n : ℤ) : Prop where
  m_range : m ∈ Ioc 0 (N : ℤ)
  n_range : n ∈ Ioc 0 (N : ℤ)
  eq_one : (n ^ 2 - m * n - m ^ 2) ^ 2 = 1

def specifiedSet : Set ℤ :=
  {k : ℤ | ∃ m : ℤ, ∃ n : ℤ, k = m ^ 2 + n ^ 2 ∧ ProblemPredicate N m n}

/- determine -/ abbrev solution : ℕ := sorry

theorem imo1981_p3 : IsGreatest (specifiedSet 1981) solution := sorry


end Imo1981P3
