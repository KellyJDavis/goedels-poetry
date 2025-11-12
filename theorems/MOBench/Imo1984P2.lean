
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1984, Problem 2

Find a pair of positive integers a and b such that

 (i) ab(a + b) is not divisible by 7.
 (ii) (a + b)⁷ - a⁷ - b⁷ is divisible by 7⁷.
-/
abbrev a : ℤ := 18
abbrev b : ℤ := 1

theorem imo1984_p2 :
    (0 < a) ∧ (0 < b) ∧
    (¬ 7 ∣ a * b * (a + b)) ∧
    7^7 ∣ (a + b)^7 - a^7 - b^7 := by sorry
