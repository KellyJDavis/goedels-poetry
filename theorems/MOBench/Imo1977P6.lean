
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1977, Problem 6

Suppose `f : ℕ+ → ℕ+` satisfies `f(f(n)) < f(n + 1)` for all `n`.
Prove that `f(n) = n` for all `n`.
-/
theorem imo1977_p6 (f : ℕ+ → ℕ+) (h : ∀ n, f (f n) < f (n + 1)) : ∀ n, f n = n := by sorry
