
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2014 N3

Consider a collection $C$ of coins, where each coin has a value of $1/n$ for some positive
integer $n$. A partition of $C$ into $N$ groups is called an *$N$-Cape Town* partition
if the total value of coins in each group is at most $1$.

Prove that if the total value of all coins in $C$ is at most $N + 1/2$,
then $C$ has an $(N + 1)$-Cape Town partition.
-/
/- special open -/ open Multiset
variable (N : ℕ) (C : Multiset ℕ)

/--
A `CapeTownPartition N C` is a partition of the multiset of coins `C` into `N + 1` groups,
where the sum of the values of the coins in each group is at most 1.
-/
structure CapeTownPartition where
  /-- The list of groups in the partition. -/
  part : Multiset (Multiset ℕ)
  /-- The number of groups is `N + 1`. -/
  card_part : card part = N + 1
  /-- The groups form a partition of `C`. -/
  sum_part : part.sum = C
  /-- The total value of coins in each group is at most 1. -/
  total_bound : ∀ G ∈ part, (G.map (fun x ↦ (x : ℚ)⁻¹)).sum ≤ 1

theorem imo_sl_2014_N3 (h_total_value : (C.map (fun x ↦ (x : ℚ)⁻¹)).sum ≤ (N : ℚ) + 1/2) :
  Nonempty (CapeTownPartition N C) := by sorry
