
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1971, Problem 5

Prove that for every natural number m there exists a finite set S of
points in the plane with the following property:
For every point s in S, there are exactly m points which are at a unit
distance from s.
-/
/- special open -/ open EuclideanGeometry






abbrev Pt := EuclideanSpace ℝ (Fin 2)

theorem imo1971_p5 (m : ℕ) :
    ∃ S : Set Pt, S.Finite ∧ ∀ s ∈ S, Nat.card {t | dist s t = 1} = m := by sorry
