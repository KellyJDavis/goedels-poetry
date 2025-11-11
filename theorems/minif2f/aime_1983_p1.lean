import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Let $x$, $y$ and $z$ all exceed $1$ and let $w$ be a positive number such that $\log_x w = 24$, $\log_y w = 40$ and $\log_{xyz} w = 12$. Find $\log_z w$. Show that it is 060.-/
theorem aime_1983_p1 (x y z w : ℕ) (ht : 1 < x ∧ 1 < y ∧ 1 < z) (hw : 0 ≤ w)
    (h0 : Real.log w / Real.log x = 24) (h1 : Real.log w / Real.log y = 40)
    (h2 : Real.log w / Real.log (x * y * z) = 12) : Real.log w / Real.log z = 60 := by sorry
