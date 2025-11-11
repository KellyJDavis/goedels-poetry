import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- $K > L > M > N$ are positive integers such that $KM + LN = (K + L - M + N)(-K + L + M + N)$. Prove that $KL + MN$ is not prime.-/
theorem imo_2001_p6 (a b c d : ℕ) (h₀ : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (h₁ : d < c) (h₂ : c < b)
    (h₃ : b < a) (h₄ : a * c + b * d = (b + d + a - c) * (b + d + c - a)) :
    ¬Nat.Prime (a * b + c * d) := by sorry
