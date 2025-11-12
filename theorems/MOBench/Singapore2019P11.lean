
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# Singapore Math Olympiad (Senior) 2019 (Round 1), Problem 11

http://www.realsutra.com/limjeck/SMO_Senior_2019.pdf

Find the value of 448 * (sin 12 degrees) * (sin 39 degrees) * (sin 51 degrees) / sin 24 degrees
-/
noncomputable abbrev solution : ℝ := 112

theorem singapore2019_r1_p11 : 448 * Real.sin (12 * Real.pi / 180) * Real.sin (39 * Real.pi / 180) * Real.sin (51 * Real.pi / 180) / Real.sin (24 * Real.pi / 180) = solution := by sorry
