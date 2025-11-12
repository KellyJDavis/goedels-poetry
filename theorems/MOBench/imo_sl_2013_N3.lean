
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2013 N3

For each positive integer $n$, let $P(n)$ denote the largest prime divisor of $n$.
Prove that there exists infinitely many $n ∈ ℕ$ such that
$$ P(n^4 + n^2 + 1) = P((n + 1)^4 + (n + 1)^2 + 1). $$
-/
noncomputable def lpf (n : ℕ) : ℕ :=
  ((Nat.primeFactors n).toList.maximum?).getD 1

theorem imo_sl_2013_N3 :
    ∀ (C : ℕ), ∃ n ≥ C, lpf (n ^ 4 + n ^ 2 + 1)
      = lpf ((n + 1) ^ 4 + (n + 1) ^ 2 + 1) := by sorry
