
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1998, Problem 6

Consider all functions f from the set of all positive integers into itself satisfying
f(t^2f(s)) = sf(t)^2 for all s and t.
Determine the least possible value of f(1998).
-/
abbrev solution : ℕ+ := 120

theorem imo1998_p6
    (f : ℕ+ → ℕ+)
    (h : ∀ s t, f (t^2 * f s) = s * (f t)^2) :
    IsLeast {n : ℕ | n = f 1998} solution := by sorry
