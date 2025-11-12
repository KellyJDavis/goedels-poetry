
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1987, Problem 6

Let $n$ be an integer greater than or equal to 2. Prove that
if $k^2 + k + n$ is prime for all integers $k$ such that
$0 <= k <= \sqrt{n/3}$, then $k^2 + k + n$ is prime for all
integers $k$ such that $0 <= k <= n - 2$.
-/
theorem imo1987_p6
    (p : ℕ)
    (h₁ : ∀ k : ℕ, k ≤ Nat.floor (Real.sqrt ((p:ℝ) / 3)) → Nat.Prime (k^2 + k + p)) :
    ∀ i ≤ p - 2, Nat.Prime (i^2 + i + p) := by sorry
