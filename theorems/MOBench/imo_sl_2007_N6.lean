
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2007 N6 (P5)

Fix $n > 1$, and let $a$ and $b$ be positive integers such that $nab - 1 ∣ (na^2 - 1)^2$.
Prove that $a = b$.
-/
/- special open -/ open Finset
abbrev bad_pair (n : ℤ) (a b : ℕ) := n * a * b - 1 ∣ (n * a ^ 2 - 1) ^ 2

theorem imo_sl_2007_N6 (hn : 1 < n) (ha : 0 < a) (hb : 0 < b) (h : bad_pair n a b) :
    a = b := by sorry
