
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2015 N1

Define the function $f : ℤ → ℤ$ by $f(n) = n ⌊n/2⌋$.
Find all integers $M$ such that $f^k(M)$ is even for some $k ∈ ℕ$.

### Notes

The original formulation is slightly different.
Instead of $f : ℤ → ℤ$, we define $f : ℚ → ℚ$ by $f(q) = q ⌊q⌋$.
Then the problem asks for which $M ∈ ℕ^+$ does there exists
  $k ∈ ℕ$ such that $f^k(M + 1/2)$ is an integer.
-/
/- special open -/ open Finset
abbrev f (n : ℤ) := n * (n / 2)

theorem imo_sl_2015_N1 : (∃ k : ℕ, 2 ∣ f^[k] M) ↔ M ≠ 3 := by sorry
