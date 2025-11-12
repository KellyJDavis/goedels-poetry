
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# Singapore Math Olympiad (Senior) 2019 (Round 1), Problem 2

http://www.realsutra.com/limjeck/SMO_Senior_2019.pdf

Simplify $(sqrt{10} - sqrt{2})^{1/3} * (sqrt{10} + sqrt{2})^{7/3}$.
-/
noncomputable abbrev solution : ℝ := 24 + 8 * √5

theorem singapore2019_r1_p2 : (√10 - √2)^(1 / 3 : ℝ) * (√10 + √2)^(7 / 3 : ℝ) = solution := by sorry
