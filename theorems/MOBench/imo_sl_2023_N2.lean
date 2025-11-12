
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2023 N2

Find all pairs $(a, p) ∈ ℕ^2$ with $a > 0$ and $p$ prime
  such that $p^a + a^4$ is a perfect square.
-/
def good (a p : ℕ) := ∃ b, p ^ a + a ^ 4 = b ^ 2

theorem imo_sl_2023_N2 {a p : ℕ} (ha : 0 < a) (hp : p.Prime) :
    good a p ↔ p = 3 ∧ (a = 1 ∨ a = 2 ∨ a = 6 ∨ a = 9) := by sorry
