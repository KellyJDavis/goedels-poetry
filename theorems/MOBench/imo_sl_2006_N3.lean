
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2006 N3

For each $n ∈ ℕ$, define
$$ f(n) = \frac{1}{n} \sum_{k = 1}^n \left\lfloor \frac{n}{k} \right\rfloor. $$
1. Prove that $f(n + 1) > f(n)$ infinitely often.
2. Prove that $f(n + 1) < f(n)$ infinitely often.
-/
/- special open -/ open Finset
def g (n : ℕ) : ℕ := (range n).sum λ k ↦ n / (k + 1)
def f (n : ℕ) : ℚ := ((g n : ℤ) : ℚ) / ((n : ℤ) : ℚ)

theorem imo_sl_2006_N3 : {n : ℕ | f n < f n.succ}.Infinite ∧ {n : ℕ | f n.succ < f n}.Infinite := by sorry
