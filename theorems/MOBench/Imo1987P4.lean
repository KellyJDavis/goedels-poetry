
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1987, Problem 4

Prove that there is no function f : ℕ → ℕ such that f(f(n)) = n + 1987
for every n.
-/
theorem imo1987_p4 : ¬∃ f : ℕ → ℕ, ∀ n, f (f n) = n + 1987 := by sorry
