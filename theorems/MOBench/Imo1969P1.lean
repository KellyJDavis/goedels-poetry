
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1969, Problem 1

Prove that there are infinitely many natural numbers a with the following property:
the number z = n⁴ + a is not prime for any natural number n.
-/
/- special open -/ open Int






theorem imo1969_p1 : Set.Infinite {a : ℕ | ∀ n : ℕ, ¬Nat.Prime (n ^ 4 + a)} := by sorry
