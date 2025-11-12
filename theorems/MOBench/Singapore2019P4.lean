
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# Singapore Math Olympiad (Senior) 2019 (Round 1), Problem 4

http://www.realsutra.com/limjeck/SMO_Senior_2019.pdf

If $\log_{21} 3 = x$, express $\log_7 9$ in terms of $x$.
-/
noncomputable abbrev solution (x : ℝ) : ℝ := 2*x / (1-x)

theorem singapore2019_r1_p4 (x : ℝ) (hx : Real.logb 21 3 = x) :
    Real.logb 7 9 = solution x := by sorry
