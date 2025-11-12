
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2020 N1

Prove that, for any positive integer $k$, there exists a prime $p$ and
  distinct elements $x_1, x_2, …, x_{k + 3} \in 𝔽_p^×$ such that for all $i ≤ k$,
$$ x_i x_{i + 1} x_{i + 2} x_{i + 3} = i. $$
-/
/- special open -/ open Function
abbrev ratSeq : ℕ → ℚ
  | 0 => 2
  | 1 => 2⁻¹
  | 2 => -4
  | 3 => -4⁻¹
  | n + 4 => (1 + (n.succ : ℚ)⁻¹) * ratSeq n

theorem imo_sl_2020_N1 (k : ℕ) :
    ∃ (p : ℕ) (_ : p.Prime) (a : Fin (k + 4) → ZMod p), a.Injective ∧ (∀ i, a i ≠ 0) ∧
        (∀ i ≤ k, a i * a (i + 1) * a (i + 2) * a (i + 3) = i.succ) := by sorry
