
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# Bulgarian Mathematical Olympiad 1998, Problem 2

A convex quadrilateral ABCD has AD = CD and ∠DAB = ∠ABC < 90°.
The line through D and the midpoint of BC intersects line AB
in point E. Prove that ∠BEC = ∠DAC. (Note: The problem is valid
without the assumption ∠ABC < 90°.)
-/
/- special open -/ open EuclideanGeometry






theorem bulgaria1998_p2
    (A B C D E M : EuclideanSpace ℝ (Fin 2))
    (H1 : dist D A = dist D C)
    (H2 : ∠ D A B = ∠ A B C)
    (H3 : M = midpoint ℝ B C) :
    ∠ B E C = ∠ D A C := by sorry
