import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- How many integer values of $x$ satisfy $|x|<3\pi$?

$\textbf{(A)} ~9 \qquad\textbf{(B)} ~10 \qquad\textbf{(C)} ~18 \qquad\textbf{(D)} ~19 \qquad\textbf{(E)} Show that it is \textbf{(D)} ~19.-/
theorem amc12b_2021_p1 (S : Finset ℤ) (h₀ : ∀ x : ℤ, x ∈ S ↔ ↑(abs x) < 3 * Real.pi) :
    S.card = 19 := by sorry
