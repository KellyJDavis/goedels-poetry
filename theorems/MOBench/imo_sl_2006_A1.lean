
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2006 A1

Let $R$ be an archimedean ring with floor.
Define the function $f : R → R$ by $$ f(x) = ⌊x⌋ (x - ⌊x⌋). $$
Prove that for any $r ∈ R$, there exists $N ∈ ℕ$ such that for all $k ≥ N$,
$$ f^{k + 2}(r) = f^k(r). $$
-/
/- special open -/ open Finset
abbrev f [LinearOrderedRing R] [FloorRing R] (r : R) := ⌈r⌉ * (r - ⌈r⌉)

theorem imo_sl_2006_A1 [LinearOrderedRing R] [FloorRing R] [Archimedean R] (r : R) :
  ∃ N, ∀ n ≥ N, f^[n + 2] r = f^[n] r := by sorry
