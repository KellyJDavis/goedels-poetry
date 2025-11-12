
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2007 A1, Part 1

Fix a linearly ordered abelian group $G$ and a positive integer $n$.
Consider a sequence $(a_i)_{i=0}^n$ of elements of $G$.

Let $(x_i)_{i=0}^n$ be a non-decreasing sequence in $G$, and let
$$ L = \max_{j \le n} |x_j - a_j|. $$
Prove that for any non-decreasing sequence $(x_i)$, the inequality $2L \ge a_k - a_m$ holds for any $k \le m \le n$.
-/
/- special open -/ open Finset
variable [LinearOrder α]

/--
The sequence `seqMax f` represents the running maximum of a sequence `f`.
`seqMax f n = max(f 0, f 1, ..., f n)`.
-/
def seqMax (f : Nat → α) : Nat → α
  | 0 => f 0
  | n + 1 => max (seqMax f n) (f n.succ)

theorem imo_sl_2007_A1a_part1 [LinearOrderedAddCommGroup G]
    (a : ℕ → G) (n : ℕ) (x : ℕ → G) (k m : ℕ)
    (h_mono : Monotone x) (h_le : k ≤ m) (h_n : m ≤ n) :
    a k - a m ≤ 2 • seqMax (λ i ↦ |x i - a i|) n := by sorry
