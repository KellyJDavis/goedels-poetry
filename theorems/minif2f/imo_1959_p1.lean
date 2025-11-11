import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Prove that the fraction $\frac{21n+4}{14n+3}$ is irreducible for every natural number $n$.-/
theorem imo_1959_p1 (n : ℕ) (h₀ : 0 < n) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by sorry
