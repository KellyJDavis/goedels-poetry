
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2001, Problem 1

Let ABC be an acute-angled triangle with O as its circumcenter. Let P
on line BC be the foot of the altitude from A. Assume that
∠BCA ≥ ∠ABC + 30°. Prove that ∠CAB + ∠COP < 90°.
-/
/- special open -/ open EuclideanGeometry






theorem imo2001_p1
    (A B C : EuclideanSpace ℝ (Fin 2))
    (hABC : AffineIndependent ℝ ![A, B, C])
    (hAcuteA : ∠ C A B < Real.pi / 2)
    (hAcuteB : ∠ A B C < Real.pi / 2)
    (hAcuteC : ∠ B C A < Real.pi / 2)
    (hAB : ∠ A B C + Real.pi / 6 ≤ ∠ B C A)
    : let ABC : Affine.Triangle _ _ := ⟨![A, B, C], hABC⟩
      let P := EuclideanGeometry.orthogonalProjection line[ℝ, B, C] A
      ∠ C A B + ∠ C ABC.circumcenter P < Real.pi / 2 := by sorry
