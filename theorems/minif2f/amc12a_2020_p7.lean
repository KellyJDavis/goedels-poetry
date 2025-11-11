import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Seven cubes, whose volumes are $1$, $8$, $27$, $64$, $125$, $216$, and $343$ cubic units, are stacked vertically to form a tower in which the volumes of the cubes decrease from bottom to top. Except for the bottom cube, the bottom face of each cube lies completely on top of the cube below it. What is the total surface area of the tower (including the bottom) in square units?

$ \textbf{(A)}\ 644\qquad\textbf{(B)}\ 658\qquad\textbf{(C)}\ 664\qquad\textbf{(D)}\ 720\qquad\textbf{(E)}\ 749 $ Show that it is \textbf{(B) }658.-/
theorem amc12a_2020_p7 (a : ℕ → ℕ) (h₀ : a 0 ^ 3 = 1) (h₁ : a 1 ^ 3 = 8) (h₂ : a 2 ^ 3 = 27)
    (h₃ : a 3 ^ 3 = 64) (h₄ : a 4 ^ 3 = 125) (h₅ : a 5 ^ 3 = 216) (h₆ : a 6 ^ 3 = 343) :
    ∑ k in Finset.range 7, 6 * ((a k) ^ 2 : ℤ) - 2 * ∑ k in Finset.range 6, (a k) ^ 2 = 658 := by sorry
