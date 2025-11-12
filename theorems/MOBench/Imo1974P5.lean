
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1974, Problem 5

What are the possible values of

 a / (a + b + d) + b / (a + b + c) + c / (b + c + d) + d / (a + c + d)

as a,b,c,d range over the positive real numbers?
-/
abbrev solution_set : Set ℝ := Set.Ioo 1 2

theorem imo1974_p5 (s : ℝ) :
    s ∈ solution_set ↔
    ∃ a b c d : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
     s = a / (a + b + d) + b / (a + b + c) +
         c / (b + c + d) + d / (a + c + d) := by sorry
