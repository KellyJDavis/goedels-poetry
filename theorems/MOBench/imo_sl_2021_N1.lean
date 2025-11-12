
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2021 N1

Find all triplets $(a, b, n)$ of positive integers such that
* $a^2 + b + 3$ is cubefree; and
* $ab + 3b + 8 = n(a^2 + b + 3)$.
-/
@[mk_iff] structure good (a b n : ℕ+) : Prop where
  cubefree : ∀ p, (a ^ 2 + b + 3).factorMultiset.count p ≤ 2
  eqn : a * b + 3 * b + 8 = n * (a ^ 2 + b + 3)

theorem imo_sl_2021_N1 :
    good a b n ↔ n = 2 ∧ ∃ k : ℕ+,
      (∀ p, (k + 2).factorMultiset.count p ≤ 1) ∧ a = k + 1 ∧ b = 2 * k := by sorry
