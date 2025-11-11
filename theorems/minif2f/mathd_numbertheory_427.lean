import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- If $A$ is the sum of the positive divisors of $500$, what is the sum of the distinct prime divisors of $A$? Show that it is 25.-/
theorem mathd_numbertheory_427 (a : ℕ) (h₀ : a = ∑ k in Nat.divisors 500, k) :
    (∑ k in Finset.filter (fun x => Nat.Prime x) (Nat.divisors a), k) = 25 := by sorry
