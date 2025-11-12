
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1981, Problem 6

Suppose that f : ℕ × ℕ → ℕ satisfies

 1) f (0, y) = y + 1
 2) f (x + 1, 0) = f (x, 1),
 3) f (x + 1, y + 1) = f (x, f (x + 1, y))

for all x y ∈ ℕ.

Determine f (4, 1981).
-/
def no_eval (x : ℕ) : ℕ := x

abbrev solution_value : ℕ := no_eval ((2^·)^[1984] 1 - 3)

theorem imo1981_p6 (f : ℕ → ℕ → ℕ)
    (h1 : ∀ y, f 0 y = y + 1)
    (h2 : ∀ x, f (x + 1) 0 = f x 1)
    (h3 : ∀ x y, f (x + 1) (y + 1) = f x (f (x + 1) y)) :
    f 4 1981 = solution_value := by sorry
