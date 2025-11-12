
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2023 N5

Let $(a_n)_{n \ge 0}$ be a strictly increasing sequence of positive integers such that:
* For all $k \ge 1$, $a_k$ divides $2(a_0 + a_1 + \dots + a_{k - 1})$.
* For each prime $p$, there exists some $k$ such that $p$ divides $a_k$.

Prove that for any positive integer $n$, there exists some $k$ such that $n$ divides $a_k$.
-/
/- special open -/ open Finset
structure GoodSeq where
  a : ℕ → ℕ
  a_strictMono : StrictMono a
  a_pos : ∀ k, 0 < a k
  a_spec : ∀ k, a k ∣ 2 * ∑ i ∈ range k, a i
  exists_dvd_a_of_prime : ∀ p : ℕ, p.Prime → ∃ k, p ∣ a k

theorem imo_sl_2023_N5 (X : GoodSeq) (N : ℕ) (hN : 0 < N) :
  ∃ k, N ∣ X.a k := by sorry
