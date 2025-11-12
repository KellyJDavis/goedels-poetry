
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1970, Problem 6

In a plane there are 100 points, no three of which are collinear.
Consider all possible triangles having these points as vertices.
Prove that no more that 70% of these triangles are acute.
-/
/- special open -/ open EuclideanGeometry






abbrev Pt := EuclideanSpace ℝ (Fin 2)

def AcuteTriangle (T : Affine.Triangle ℝ Pt) : Prop :=
  ∠ (T.points 1) (T.points 2) (T.points 3) < Real.pi / 2 ∧
  ∠ (T.points 2) (T.points 3) (T.points 1) < Real.pi / 2 ∧
  ∠ (T.points 3) (T.points 1) (T.points 2) < Real.pi / 2

theorem imo1970_p6
    (P : Fin 100 → Pt)
    (hP : ∀ a b c : Fin 100,
             List.Nodup [a, b, c] → ¬ Collinear ℝ {P c, P b, P c}) :
    let cardAll := Nat.card { t : Affine.Triangle ℝ Pt |
                              ∃ a b c : Fin 100, ![P a, P b, P c] = t.points }
    let cardAcute :=
      Nat.card { t : Affine.Triangle ℝ Pt | ∃ a b c : Fin 100, ![P a, P b, P c] = t.points ∧
                                            AcuteTriangle t }
    (cardAcute : ℚ) / cardAll ≤ 7 / 10 := by sorry
