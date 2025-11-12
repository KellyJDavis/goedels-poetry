
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# USA Mathematical Olympiad 1992, Problem 2

Prove that

1 / cos 0° / cos 1° + 1 / cos 1° / cos 2° + ... + 1 / cos 88° / cos 99° = cos 1° / sin² 1°
-/
theorem usa1992_p2 :
  ∑ i ∈ Finset.range 89, 1 / Real.cos (i * Real.pi / 180) / Real.cos ((i + 1) * Real.pi / 180) =
  Real.cos (Real.pi / 180) / Real.sin (Real.pi / 180) ^ 2 := by sorry
