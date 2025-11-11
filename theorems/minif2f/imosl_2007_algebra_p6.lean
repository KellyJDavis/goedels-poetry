import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- For a series $\{a_n\}$, we have $\sum_{n=0}^{99} a_{n+1}^2 = 1$. Show that $\sum_{n=0}^{98} (a_{n+1}^2 a_{n+2}) + a_{100}^2 * a_1 < \frac{12}{25}$.-/
theorem imosl_2007_algebra_p6 (a : ℕ → NNReal) (h₀ : (∑ x in Finset.range 100, a (x + 1) ^ 2) = 1) :
    (∑ x in Finset.range 99, a (x + 1) ^ 2 * a (x + 2)) + a 100 ^ 2 * a 1 < 12 / 25 := by sorry
