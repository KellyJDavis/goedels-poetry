
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2018 N5

Determine whether there exists $x, y, z, t ∈ ℕ^+$ such that
  $xy - zt = x + y = z + t$ and both $xy$ and $zt$ are perfect squares.
-/
/- special open -/ open Finset
def good (v : Fin 4 → ℤ) := v 0 * v 1 - v 2 * v 3 = v 0 + v 1 ∧ v 0 + v 1 = v 2 + v 3

variable (hv : good v)

theorem imo_sl_2018_N5 (hv0 : ∀ i, v i ≠ 0) :
    ¬((∃ x, v 0 * v 1 = x ^ 2) ∧ ∃ y, v 2 * v 3 = y ^ 2) := by sorry
