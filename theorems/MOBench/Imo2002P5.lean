
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2002, Problem 5

Determine all functions f : ℝ → ℝ such that

  (f(x) + f(z))(f(y) + f(t)) = f(xy - zt) + f(xt + yz)

for all real numbers x,y,z,t.
-/
abbrev SolutionSet : Set (ℝ → ℝ) :=
  { fun x ↦ 0, fun x ↦ 1/2, fun x ↦ x^2 }

theorem imo2002_p5 (f : ℝ → ℝ) :
    f ∈ SolutionSet ↔
    ∀ x y z t, (f x + f z) * (f y + f t) =
               f (x * y - z * t) + f (x * t + y * z) := by sorry
