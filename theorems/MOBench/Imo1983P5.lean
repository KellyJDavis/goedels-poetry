
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1983, Problem 5

Is it possible to choose $1983$ distinct positive integers,
all less than or equal to $10^5$,
no three of which are consecutive terms of an arithmetic progression?
Justify your answer.
-/
theorem imo1983_p5 :
  ∃ S : Finset ℕ, S.card = 1983 ∧
  (∀ x ∈ S, x ≤ 10^5) ∧
  ∀ x ∈ S, ∀ y ∈ S, ∀ z ∈ S, x < y ∧ y < z → x + z ≠ 2 * y := by sorry
