import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- For every real number $x_1$, construct the sequence $x_1,x_2,\ldots$ by setting $x_{n+1}=x_n \left(x_n + \frac{1}{n}\right)$ for each $n \geq 1$.  Prove that there exists exactly one value of $x_1$ for which $0<x_n<x_{n+1}<1$ for every $n$.-/
theorem imo_1985_p6 (f : ℕ → NNReal → ℝ) (h₀ : ∀ x, f 1 x = x)
    (h₁ : ∀ x n, f (n + 1) x = f n x * (f n x + 1 / n)) :
    ∃! a, ∀ n, 0 < n → 0 < f n a ∧ f n a < f (n + 1) a ∧ f (n + 1) a < 1 := by sorry
