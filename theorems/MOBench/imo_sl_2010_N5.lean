
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2010 N5 (P3)

Given $c ∈ ℕ$, find all functions $f : ℕ → ℕ$ such that
  $(f(m) + n + c)(f(n) + m + c)$ is a square for all $m, n ∈ ℕ$.
-/
def good (c : ℕ) (f : ℕ → ℕ) := ∀ m n, ∃ k, (f m + n + c) * (f n + m + c) = k ^ 2

variable (hp : Nat.Prime p) (h : ∃ k : ℕ, a * b = k ^ 2)

theorem imo_sl_2010_N5 : good c f ↔ ∃ k, f = (· + k) := by sorry
