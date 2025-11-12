
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
There are 101 positive integers arranged in a circle.
Suppose that the integers sum to 300.
Prove that there exists a contiguous subarray that sums to 200.

https://mathstodon.xyz/@alexdbolton/110292738044661739
https://math.stackexchange.com/questions/282589/101-positive-integers-placed-on-a-circle
-/
theorem integers_in_a_circle
    (a : ZMod 101 → ℤ)
    (ha : ∀ i, 1 ≤ a i)
    (ha_sum : ∑ i : ZMod 101, a i = 300)
    : ∃ j : ZMod 101, ∃ n : ℕ, ∑ i ∈ Finset.range n, a (j + i) = 200 := by sorry
