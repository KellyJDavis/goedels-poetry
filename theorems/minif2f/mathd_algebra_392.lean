import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- The sum of the squares of three consecutive positive even numbers is $12296$. Find the product of the three numbers divided by $8$. Show that it is 32736.-/
theorem mathd_algebra_392 (n : ℕ) (h₀ : Even n)
    (h₁ : (↑n - 2) ^ 2 + ↑n ^ 2 + (↑n + 2) ^ 2 = (12296 : ℤ)) :
    (↑n - 2) * ↑n * (↑n + 2) / 8 = (32736 : ℤ) := by sorry
