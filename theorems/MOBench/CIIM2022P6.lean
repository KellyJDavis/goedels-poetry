
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# Iberoamerican Interuniversity Mathematics Competition 2022, Problem 6

Given a positive integer m, let d(m) be the number of postive
divisors of m. Show that for every positive integer n, one
has
       d((n + 1)!) ≤ 2d(n!).
-/
def d : ℕ → ℕ
| m => (Nat.divisors m).card

theorem ciim2022_p6 (n : ℕ) (hn : 0 < n) :
    d (Nat.factorial (n + 1)) ≤ 2 * d (Nat.factorial n) := by sorry
