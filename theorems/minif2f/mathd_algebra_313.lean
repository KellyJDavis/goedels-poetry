import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Complex numbers are often used when dealing with alternating current (AC) circuits. In the equation $V = IZ$, $V$ is voltage, $I$ is current, and $Z$ is a value known as impedance. If $V = 1+i$ and $Z=2-i$, find $I$. Show that it is \frac{1}{5} + \frac{3}{5}i.-/
theorem mathd_algebra_313 (v i z : ℂ) (h₀ : v = i * z) (h₁ : v = 1 + Complex.I)
    (h₂ : z = 2 - Complex.I) : i = 1 / 5 + 3 / 5 * Complex.I := by sorry
