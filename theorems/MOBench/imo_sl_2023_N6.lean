
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2023 N6

A sequence $(a_n)_{n ≥ 0}$ is called *kawaii* if $a_0 = 0$, $a_1 = 1$, and for any $n ∈ ℕ$,
$$ a_{n + 2} + 2 a_n = 3 a_{n + 1} \text{ or } a_{n + 2} + 3 a_n = 4 a_{n + 1}. $$
A non-negative integer $m$ is said to be *kawaii* if it belongs to some kawaii sequence.

Let $m ∈ ℕ$ such that both $m$ and $m + 1$ are kawaii.
Prove that $3 ∣ m$ and $m/3$ belongs to a kawaii sequence.
-/
@[ext] structure KawaiiSequence (S : Set ℕ) where
  a : ℕ → ℕ
  a_zero : a 0 = 0
  a_one : a 1 = 1
  a_step : ∀ n, ∃ c : S, a (n + 2) + c * a n = (c + 1) * a (n + 1)

theorem imo_sl_2023_N6 (hn : ∃ (X : KawaiiSequence {2, 3}) (k : ℕ), n = X.a k)
    (hn0 : ∃ (X : KawaiiSequence {2, 3}) (k : ℕ), n + 1 = X.a k) :
    ∃ m, (∃ (X : KawaiiSequence {2, 3}) (k : ℕ), m = X.a k) ∧ n = 3 * m := by sorry
