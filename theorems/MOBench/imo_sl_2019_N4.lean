
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2019 N4

Fix some $C ∈ ℕ$.
Find all functions $f : ℕ → ℕ$ such that $a + f(b) ∣ a^2 + b f(a)$
  for any $a, b ∈ ℕ$ satisfying $a + b > C$.
-/
/- special open -/ open List
def goodPNat (C : ℕ+) (f : ℕ+ → ℕ+) :=
  ∀ a b : ℕ+, C < a + b → a + f b ∣ a ^ 2 + b * f a

theorem imo_sl_2019_N4 : goodPNat C f ↔ ∃ k : ℕ+, f = (k * ·) := by sorry
