
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
(From Mathematical Puzzles: A Connoisseur's Collection by Peter Winkler.)

Let n be a natural number. Prove that

  (a) n has a (nonzero) multiple whose representation in base 10 contains
      only zeroes and ones; and
-/
theorem zeroes_and_ones
    (n : ℕ) : ∃ k : ℕ+, ∀ e ∈ Nat.digits 10 (n * k), e = 0 ∨ e = 1 := by sorry
