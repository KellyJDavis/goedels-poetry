
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2021 C2

Fix some positive integer $n$, and denote $[n] = \{0, 1, …, n - 1\}$.
Find all positive integers $m ∈ ℕ$ such that there exists a
  function $f : ℤ/mℤ → [n]$ with the following property:
  for any $k ∈ ℤ/mℤ$ and $i ∈ [n]$, there exists $j ≤ n$ such that $f(k + j) = i$.
-/
/- special open -/ open Finset
def good (f : Fin (m + 1) → Fin n) := ∀ k i, ∃ j ≤ n, f (k + j) = i

theorem imo_sl_2021_C2 {n m : ℕ} :
    (∃ f : Fin m.succ → Fin n.succ, good f) ↔ m.succ % n.succ ≤ m.succ / n.succ := by sorry
