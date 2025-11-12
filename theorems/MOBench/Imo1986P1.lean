
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1986, Problem 1

Let d be any positive integer not equal to 2, 5 or 13.
Show that one can find distinct a, b in the set {2, 5, 13, d} such that ab - 1
is not a perfect square.
-/
theorem imo1986_p1 (d : ℤ) (_hdpos : 1 ≤ d) (h2 : d ≠ 2) (h5 : d ≠ 5) (h13 : d ≠ 13) :
    ∃ a b :({2, 5, 13, d} : Finset ℤ), (a ≠ b) ∧ ¬ ∃ z, z^2 = (a * (b : ℤ) - 1) := by sorry
