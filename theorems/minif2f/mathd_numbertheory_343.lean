import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- What is the units digit of the product of all of the odd integers between 0 and 12? Show that it is 5.-/
theorem mathd_numbertheory_343 : (∏ k in Finset.range 6, (2 * k + 1)) % 10 = 5 := by sorry
