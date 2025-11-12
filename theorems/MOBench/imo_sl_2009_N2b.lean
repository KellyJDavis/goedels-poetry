
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2009 N2

For each positive integer $n$, let $Ω(n)$ denote the number of
  prime factors of $n$, counting multiplicity.
For convenience, we denote $Ω(0) = 0$.
2. Prove that for any $a, b ∈ ℕ$, if $Ω((a + k)(b + k))$ is even
    for all $k ∈ ℕ$, then $a = b$.
-/
/- special open -/ open ArithmeticFunction
theorem imo_sl_2009_N2b_part2 (h : ∀ k, Even (Ω ((a + k) * (b + k)))) : a = b := by sorry
