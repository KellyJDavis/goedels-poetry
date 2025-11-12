
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2009 N3

Let $f : ℕ → ℤ$ be a non-constant function such that
  $a - b ∣ f(a) - f(b)$ for any $a, b ∈ ℕ$.
Prove that there exists infinitely many primes $p$
  that divide $f(c)$ for some $c ∈ ℕ$.

### Notes

In this file, the infinitude of such primes is rephrased as follows:
  for any $k ∈ ℕ$, there exists a prime $p ≥ k$ such that
  $p ∣ f(c)$ for some $c ∈ ℕ$.
The equivalence is clear, and this avoids importing `Mathlib.Data.Set.Finite`.
-/
variable {f : ℕ → ℤ} (h : ∀ a b : ℕ, (a : ℤ) - b ∣ f a - f b)

theorem imo_sl_2009_N3 (h0 : ∀ C : ℤ, ∃ b : ℕ, f b ≠ C) (K : ℕ) :
    ∃ p : ℕ, K ≤ p ∧ p.Prime ∧ ∃ c, (p : ℤ) ∣ f c := by sorry
