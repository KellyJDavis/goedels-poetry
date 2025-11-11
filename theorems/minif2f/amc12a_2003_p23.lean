import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- How many perfect squares are divisors of the product $1! \cdot 2! \cdot 3! \cdot \hdots \cdot 9!$?

$ \textbf{(A)}\ 504\qquad\textbf{(B)}\ 672\qquad\textbf{(C)}\ 864\qquad\textbf{(D)}\ 936\qquad\textbf{(E)}\ 1008 $ Show that it is \mathrm{(B)}.-/
theorem amc12a_2003_p23 (S : Finset ℕ)
    (h₀ : ∀ k : ℕ, k ∈ S ↔ 0 < k ∧ (k * k : ℕ) ∣ ∏ i in Finset.Icc 1 9, i !) : S.card = 672 := by sorry
