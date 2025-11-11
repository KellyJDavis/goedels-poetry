import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Let $a,b,c,d$ be odd integers such that $0<a<b<c<d$ and $ad=bc$. Prove that if $a+d=2^k$ and $b+c=2^m$ for some integers $k$ and $m$, then $a=1$.-/
theorem imo_1984_p6 (a b c d k m : ℕ) (h₀ : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
    (h₁ : Odd a ∧ Odd b ∧ Odd c ∧ Odd d) (h₂ : a < b ∧ b < c ∧ c < d) (h₃ : a * d = b * c)
    (h₄ : a + d = 2 ^ k) (h₅ : b + c = 2 ^ m) : a = 1 := by sorry
