
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# Singapore Math Olympiad (Senior) 2019 (Round 1), Problem 7

http://www.realsutra.com/limjeck/SMO_Senior_2019.pdf

Suppose that $\tan x = 5$. Find the value of $\frac{6 + \sin 2x}{1 + \cos 2x}$.
-/
noncomputable abbrev solution : ℝ := 83

theorem singapore2019_r1_p7 (x : ℝ) (hx : tan x = 5) :
    (6 + sin (2 * x)) / (1 + cos (2 * x)) = solution := by sorry
