
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1981, Problem 3

Determine the maximum value of `m ^ 2 + n ^ 2`, where `m` and `n` are integers in
`{1, 2, ..., 1981}` and `(n ^ 2 - m * n - m ^ 2) ^ 2 = 1`.
-/
/- special open -/ open Int Set






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

abbrev solution : ℕ := 3524578

theorem imo1981_p3 : IsGreatest (specifiedSet 1981) solution := by sorry
