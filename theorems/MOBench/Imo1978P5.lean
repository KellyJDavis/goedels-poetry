
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1978, Problem 5

Let a_k be a sequence of distinct positive integers for k = 1,2,3, ...

Prove that for all natral numbers n, we have:

sum_{k=1}^{n} a(k)/(k^2) >= sum_{k=1}^{n} (1/k).
-/
/- special open -/ open Finset







theorem imo_1978_p5
  (n : ℕ)
  (f : ℕ → ℕ)
  (h₀ : ∀ (m : ℕ), 0 < m → 0 < f m)
  (h₁ : ∀ (p q : ℕ), 0 < p → 0 < q → p ≠ q → f p ≠ f q)
  (h₂ : 0 < n) :
  (∑ k ∈ Finset.Icc 1 n, (1 : ℝ) / k) ≤ ∑ k ∈ Finset.Icc 1 n, ((f k):ℝ) / k ^ 2 := by sorry
