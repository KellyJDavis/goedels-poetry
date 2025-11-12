
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# USA Mathematical Olympiad 1978, Problem 1

Given that a,b,c,d,e are real numbers such that

  a  + b  + c  + d  + e  = 8
  a² + b² + c² + d² + e² = 16,

determine the maximum value of e.
-/
noncomputable abbrev max_e : ℝ := (16 : ℝ) / 5

abbrev IsGood (a b c d e : ℝ) : Prop :=
  a + b + c + d + e = 8 ∧ a^2 + b^2 + c^2 + d^2 + e^2 = 16

theorem usa1978_p1 :
    IsGreatest { e : ℝ | ∃ a b c d : ℝ, IsGood a b c d e } max_e := by sorry
