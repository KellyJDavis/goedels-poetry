
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# USA Mathematical Olympiad 1983, Problem 5

Consider an open interval of length 1/2 on the real number line, where
n is a positive integer. Prove that the number of irreducible fractions
p/q, with 1 ≤ q ≤ n, contained in the given interval is at most (n + 1) / 2.
-/
theorem usa1983_p5 (x : ℝ) (n : ℕ) (hn : 0 < n) :
    let fracs := { q : ℚ | q.den ≤ n ∧ ↑q ∈ Set.Ioo x (x + 1 / n)}
    fracs.Finite ∧ fracs.ncard ≤ (n + 1) / 2 := by sorry
