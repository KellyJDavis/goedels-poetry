import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Suppose that the roots of the polynomial $P(x)=x^3+ax^2+bx+c$ are $\cos \frac{2\pi}7,\cos \frac{4\pi}7,$ and $\cos \frac{6\pi}7$, where angles are in radians. What is $abc$?

$\textbf{(A) }{-}\frac{3}{49} \qquad \textbf{(B) }{-}\frac{1}{28} \qquad \textbf{(C) }\frac{\sqrt[3]7}{64} \qquad \textbf{(D) }\frac{1}{32}\qquad \textbf{(E) }\frac{1}{28}$ Show that it is \textbf{(D) }\frac{1}{32}.-/
theorem amc12a_2021_p22 (a b c : ℝ) (f : ℝ → ℝ) (h₀ : ∀ x, f x = x ^ 3 + a * x ^ 2 + b * x + c)
    (h₁ :
      f ⁻¹' {0} =
        {Real.cos (2 * Real.pi / 7), Real.cos (4 * Real.pi / 7), Real.cos (6 * Real.pi / 7)}) :
    a * b * c = 1 / 32 := by sorry
