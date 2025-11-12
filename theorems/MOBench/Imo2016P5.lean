
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2016, Problem 5

The equation

  (x - 1)(x - 2) ... (x - 2016) = (x - 1)(x - 2) ... (x - 2016)

is written on the board. What is the least possible value of k
for which it is possible to erase exactly k of these 4032 factors
such that at least one factor remains on each side and the resulting
equation has no real solutions?
-/
abbrev solution_value : ℕ := 2016

theorem imo2015_p5 :
    IsLeast { k | ∃ L R : Finset ℕ,
                  L ⊂ Finset.Icc 1 2016 ∧
                  R ⊂ Finset.Icc 1 2016 ∧
                  L.card + R.card = k ∧
                  ¬∃ x : ℝ,
                   ∏ i ∈ Finset.Icc 1 2016 \ L, (x - (i : ℝ)) =
                   ∏ i ∈ Finset.Icc 1 2016 \ R, (x - (i : ℝ)) }
            solution_value := by sorry
