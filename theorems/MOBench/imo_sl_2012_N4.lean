
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2012 N4

An integer $a$ is called *friendly* if there exist positive integers $m, n$ such that
$$ (m^2 + n)(n^2 + m) = a(m - n)^3. $$

1. Prove that $\{1, 2, \dots, 2012\}$ contains at least $500$ friendly integers.
2. Is $2$ friendly?
-/
/- special open -/ open Finset Classical
/-- An integer `a` is friendly if it satisfies the given condition for some positive integers `m` and `n`. -/
def IsFriendly (a : ℤ) : Prop :=
  ∃ m > 0, ∃ n > 0, (m ^ 2 + n) * (n ^ 2 + m) = a * (m - n) ^ 3

theorem imo_sl_2012_N4 :
  500 ≤ ((Icc 1 2012).filter (IsFriendly)).card ∧
  ¬ IsFriendly 2 := by sorry
