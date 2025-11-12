
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2018 N4 (P5)

Let $(a_n)_{n ≥ 1}$ be a sequence of positive integers such that for $n$ large enough,
$$ \frac{a_1}{a_2} + \frac{a_2}{a_3} + … + \frac{a_{n - 1}}{a_n} + \frac{a_n}{a_1} ∈ ℤ. $$
Prove that $(a_n)_{n ≥ 1}$ is eventually constant.
-/
/- special open -/ open Finset
theorem imo_sl_2018_N4 {a : ℕ → ℕ} (ha : ∀ n, 0 < a n)
    (ha0 : ∃ K, ∀ n ≥ K, ∃ z : ℤ, (z : ℚ) =
      (∑ i ∈ Finset.range n, (a i : ℚ) / a (i + 1)) + a n / a 0) :
    ∃ C N, ∀ n, a (n + N) = C := by sorry
