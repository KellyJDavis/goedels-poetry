
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2017 N1 (P1)

For each $n ∈ ℕ$, define $f(n)$ by $\sqrt{n}$ if $n$ is a square and $n + 3$ otherwise.
Find all $N ∈ ℕ$ such that $\{n : f^n(N) = a\}$ is infinite for some $a ∈ ℕ$.

-/
/- special open -/ open Finset
def f (p k : ℕ) : ℕ := if k.sqrt ^ 2 = k then k.sqrt else k + p

def good (p N : ℕ) := ∃ a, ∀ m, ∃ n ≥ m, (f p)^[n] N = a

theorem imo_sl_2017_N1 : good 3 N ↔ N = 1 ∨ 3 ∣ N := by sorry
