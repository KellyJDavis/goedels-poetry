
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2009 A6 (P3)

Let $f : ℕ → ℕ$ be a strictly increasing function.
Suppose that there exists $A, B, C, D ∈ ℕ$ such that
  $f(f(n)) = An + B$ and $f(f(n) + 1) = Cn + D$ for any $n ∈ ℕ$.
Prove that there exists $M, N ∈ ℕ$ such that $f(n) = Mn + N$ for all $n ∈ ℕ$.
-/
theorem imo_sl_2009_A6 {f : ℕ → ℕ} (hf : StrictMono f)
    (h : ∃ A B, ∀ n, f (f n) = A * n + B) (h0 : ∃ C D, ∀ n, f (f n + 1) = C * n + D) :
    ∃ M N, ∀ n, f n = M * n + N := by sorry
