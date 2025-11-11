import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Let $f(n)$ be a function $f: \mathbb{N}^{+}\to\mathbb{N}^{+}$. Prove that if $ f(n+1) > f(f(n)) $ for each positive integer $n$, then $f(n)=n$.-/
theorem imo_1977_p6 (f : ℕ → ℕ) (h₀ : ∀ n, 0 < f n) (h₁ : ∀ n, 0 < n → f (f n) < f (n + 1)) :
    ∀ n, 0 < n → f n = n := by sorry
