
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
(From Mathematical Puzzles: A Connoisseur's Collection by Peter Winkler.)

Let n be a natural number. Prove that

  (b) 2^n has a multiple whose representation contains only ones and twos.
-/
theorem ones_and_twos
    (n : ℕ) : ∃ k : ℕ+, ∀ e ∈ Nat.digits 10 (2^n * k), e = 1 ∨ e = 2 := by sorry
