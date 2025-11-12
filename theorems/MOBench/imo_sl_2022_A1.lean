
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2022 A1

Let $R$ be a totally ordered ring.
Let $(a_n)_{n ≥ 0}$ be a sequence of non-negative elements of $R$ such that for any $n ∈ ℕ$,
$$ a_{n + 1}^2 + a_n a_{n + 2} ≤ a_n + a_{n + 2}. $$
Show that $a_N ≤ 1$ for all $N ≥ 2$.
-/
variable [LinearOrderedRing R]

theorem imo_sl_2022_A1 {a : ℕ → R} (h : ∀ i, 0 ≤ a i)
    (h0 : ∀ i, a (i + 1) ^ 2 + a i * a (i + 2) ≤ a i + a (i + 2))
    (N : ℕ) (h1 : 2 ≤ N) : a N ≤ 1 := by sorry
