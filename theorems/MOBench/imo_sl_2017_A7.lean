
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2017 A7

Let $(b_n)_{n \ge 0}$ be a sequence of positive integers.
Let $(a_n)_{n \ge 0}$ be a sequence of integers defined by $a_0 = 0$, $a_1 = 1$, and
for $n \ge 0$:
- $a_{n + 2} = a_{n + 1} b_{n + 1} + a_n$ if $b_n = 1$;
- $a_{n + 2} = a_{n + 1} b_{n + 1} - a_n$ if $b_n > 1$.

Prove that $\max\{a_n, a_{n + 1}\} \ge n$ for any $n \ge 0$.
-/
def a (b : ℕ → ℤ) : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | n + 2 => a b (n + 1) * b (n + 1) + a b n * if b n = 1 then 1 else -1

theorem imo_sl_2017_A7 (b : ℕ → ℤ) (b_pos : ∀ n, 0 < b n) (n : ℕ) :
  (n : ℤ) ≤ a b n ∨ (n : ℤ) ≤ a b (n + 1) := by sorry
