
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2021 N2 (P1)

Let $n ≥ 99$ be an integer.
The non-negative integers are coloured using two colours.
Prove that there exists $a, b ∈ ℕ$ of the same colour such that
  $n ≤ a < b ≤ 2n$ and $a + b$ is a square.
-/
def good (n : ℕ) :=
  ∀ x : ℕ → Bool, ∃ a b, n ≤ a ∧ a < b ∧ b ≤ 2 * n ∧ x a = x b ∧ ∃ k, a + b = k ^ 2

theorem imo_sl_2021_N2 (h : 99 ≤ n) : good n := by sorry
