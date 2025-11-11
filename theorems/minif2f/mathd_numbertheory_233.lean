import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Find $24^{-1} \pmod{11^2}$. That is, find the residue $b$ for which $24b \equiv 1\pmod{11^2}$.

Express your answer as an integer from $0$ to $11^2-1$, inclusive. Show that it is 116.-/
theorem mathd_numbertheory_233 (b : ZMod (11 ^ 2)) (h₀ : b = 24⁻¹) : b = 116 := by sorry
