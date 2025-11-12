
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2007 A1, Part 2

Fix a linearly ordered abelian group $G$ and a positive integer $n$.
Consider a sequence $(a_i)_{i=0}^n$ of elements of $G$.
Let $L(x) = \max_{j \le n} |x_j - a_j|$ for a non-decreasing sequence $(x_i)$.

Prove that for any $g \in G$ such that $2g \ge a_k - a_m$ for any $k \le m \le n$,
there exists a non-decreasing sequence $(x_i)$ such that $L(x) \le g$.
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


theorem imo_sl_2007_A1b_part2 [LinearOrderedAddCommGroup G]
    (a : ℕ → G) (n : ℕ) (g : G)
    (h_g : ∀ k m : ℕ, k ≤ m → m ≤ n → a k - a m ≤ 2 • g) :
    ∃ x : ℕ → G, Monotone x ∧ seqMax (λ i ↦ |x i - a i|) n ≤ g := by sorry
