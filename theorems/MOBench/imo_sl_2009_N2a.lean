
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2009 N2

For each positive integer $n$, let $Ω(n)$ denote the number of
  prime factors of $n$, counting multiplicity.
For convenience, we denote $Ω(0) = 0$.
1. Prove that for any $N ∈ ℕ$, there exists $a, b ∈ ℕ$ distinct
    such that $Ω((a + k)(b + k))$ is even for all $k < N$.
-/
/- special open -/ open ArithmeticFunction
theorem imo_sl_2009_N2a_part1 (N) : ∃ a b, a ≠ b ∧ ∀ k < N, Even (Ω ((a + k) * (b + k))) := by sorry
