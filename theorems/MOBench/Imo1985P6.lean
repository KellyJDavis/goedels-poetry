
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1985, Problem 6

For every real number x_1, construct the sequence {x_1,x_2, ...}
by setting x_{n+1} = x_n * (x_n + 1 / n) for each n >= 1.


Prove that there exists exactly one value of x_1 for which
0 < x_n , x_n < x_{n+1}, and x_{n+1} < 1 for every n.
-/
theorem imo_1985_p6
  (f : ℕ → ℝ → ℝ)
  (h₀ : ∀ x, f 1 x = x)
  (h₁ : ∀ n x, 0 < n → f (n + 1) x = f n x * (f n x + 1 / n)) :
  ∃! a, ∀ n, 0 < n → 0 < f n a ∧ f n a < f (n + 1) a ∧ f (n + 1) a < 1 := by sorry
