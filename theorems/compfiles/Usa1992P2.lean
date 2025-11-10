import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Real.Pi.Bounds
import Mathlib.Tactic

/-!
# USA Mathematical Olympiad 1992, Problem 2

Prove that

1 / cos 0° / cos 1° + 1 / cos 1° / cos 2° + ... + 1 / cos 88° / cos 99° = cos 1° / sin² 1°
-/

namespace Usa1992P2

open Real

theorem usa1992_p2 :
  ∑ i ∈ Finset.range 89, 1 / cos (i * π / 180) / cos ((i + 1) * π / 180) =
  cos (π / 180) / sin (π / 180) ^ 2 := sorry


end Usa1992P2
