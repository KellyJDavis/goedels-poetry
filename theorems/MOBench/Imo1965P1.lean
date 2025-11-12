
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1965, Problem 1

Determine all values x in the interval 0 ≤ x ≤ 2π which
satisfy the inequality

   2 cos x ≤ |√(1 + sin 2x) − √(1 − sin 2x)| ≤ √2.
-/
/- special open -/ open Set




abbrev the_answer : Set ℝ := Set.Icc (Real.pi/4) (7*Real.pi/4)

theorem imo1965_p1 :
    {x ∈ Set.Icc 0 (2*Real.pi) |
       |Real.sqrt (1 + Real.sin (2*x)) - Real.sqrt (1 - Real.sin (2*x))| ∈ Set.Icc (2 * Real.cos x) (Real.sqrt 2)}
     = the_answer := by sorry
