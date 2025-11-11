import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Let $p, q, r$ be integers with $1 < p < q < r$. Show that if $(p - 1) (q - 1) (r - 1)$ divides $pqr - 1$, then $(p, q, r) = (2, 4, 8)$ or $(p, q, r) = (3, 5, 15)$.-/
theorem imo_1992_p1 (p q r : ℤ) (h₀ : 1 < p ∧ p < q ∧ q < r)
    (h₁ : (p - 1) * (q - 1) * (r - 1) ∣ p * q * r - 1) :
    (p, q, r) = (2, 4, 8) ∨ (p, q, r) = (3, 5, 15) := by sorry
