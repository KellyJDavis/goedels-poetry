import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Find the value of $a_2+a_4+a_6+a_8+\ldots+a_{98}$ if $a_1$, $a_2$, $a_3\ldots$ is an [[arithmetic progression]] with common difference 1, and $a_1+a_2+a_3+\ldots+a_{98}=137$. Show that it is 093.-/
theorem aime_1984_p1 (u : ℕ → ℚ) (h₀ : ∀ n, u (n + 1) = u n + 1)
    (h₁ : (∑ k in Finset.range 98, u k.succ) = 137) :
    (∑ k in Finset.range 49, u (2 * k.succ)) = 93 := by sorry
