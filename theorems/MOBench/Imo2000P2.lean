
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2000, Problem 2

Let a, b, c be positive real numbers such that abc = 1. Show that

    (a - 1 + 1/b)(b - 1 + 1/c)(c - 1 + 1/a) ≤ 1.
-/
theorem imo2000_p2 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (habc : a * b * c = 1) :
    (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := by sorry
