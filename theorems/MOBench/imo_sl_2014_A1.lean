
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2014 A1

Let $(z_n)_{n \ge 0}$ be an infinite sequence of positive integers.
1. Prove that there exists a unique non-negative integer $N$ such that
   $$ N z_N < \sum_{j = 0}^N z_j \le (N + 1) z_{N + 1}. $$
   (Note: The sum is often written as $\sum_{j=0}^{N-1} z_j \le N z_N < \sum_{j=0}^N z_j$. The version
   above is what is used in the formalization).
2. Prove that $N$ is positive.
3. Show that $\binom{N}{2} < z_0$.
-/
/- special open -/ open Finset
variable {z : ℕ → ℤ}

/-- `IsTheN z N` is the property that `N` satisfies the double inequality from the problem. -/
def IsTheN (z : ℕ → ℤ) (N : ℕ) : Prop :=
  (N : ℤ) * z N < (∑ i in range (N + 1), z i) ∧
  (∑ i in range (N + 1), z i) ≤ (N + 1) * z (N + 1)

theorem imo_sl_2014_A1 (hz_pos : ∀ n, 0 < z n) (hz_mono : StrictMono z) :
  (∃! N, IsTheN z N) ∧
  (∀ N, IsTheN z N → 0 < N ∧ N.choose 2 < (z 0).natAbs) := by sorry
