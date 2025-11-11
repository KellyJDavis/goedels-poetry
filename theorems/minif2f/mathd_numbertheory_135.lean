import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Let $n = 3^{17} + 3^{10}$. It is known that $11$ divides into $n+1$. If $n$ can be written in base $10$ as $ABCACCBAB$, where $A,B,C$ are distinct digits such that $A$ and $C$ are odd and $B$ is not divisible by $3$, find $100A + 10B + C$. Show that it is 129.-/
theorem mathd_numbertheory_135 (n A B C : ℕ) (h₀ : n = 3 ^ 17 + 3 ^ 10) (h₁ : 11 ∣ n + 1)
    (h₂ : [A, B, C].Pairwise (· ≠ ·)) (h₃ : {A, B, C} ⊂ Finset.Icc 0 9) (h₄ : Odd A ∧ Odd C)
    (h₅ : ¬3 ∣ B) (h₆ : Nat.digits 10 n = [B, A, B, C, C, A, C, B, A]) :
    100 * A + 10 * B + C = 129 := by sorry
