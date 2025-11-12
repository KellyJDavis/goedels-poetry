
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
USA Mathematical Olympiad 2000, Problem 1

A function f : ℝ → ℝ is called "very convex" if it satisfies

  ∀ x y : ℝ, (f(x) + f(y))/2 ≥  f((x + y)/2) + |x - y|.

Show that there exist no very convex functions.
-/
theorem usa2000_p1 :
    ¬∃ f : ℝ → ℝ,
      ∀ x y : ℝ, f ((x + y) / 2) + |x - y| ≤ (f x + f y) / 2 := by sorry
