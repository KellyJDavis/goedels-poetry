import Mathlib.Data.Nat.ModEq
import Mathlib.Data.Nat.Digits

/-!
(From Mathematical Puzzles: A Connoisseur's Collection by Peter Winkler.)

Let n be a natural number. Prove that

  (a) n has a (nonzero) multiple whose representation in base 10 contains
      only zeroes and ones; and
  (b) 2^n has a multiple whose representation contains only ones and twos.
-/

namespace ZerosOnesAndTwos

theorem zeroes_and_ones
    (n : ℕ) : ∃ k : ℕ+, ∀ e ∈ Nat.digits 10 (n * k), e = 0 ∨ e = 1 := sorry

theorem ones_and_twos
    (n : ℕ) : ∃ k : ℕ+, ∀ e ∈ Nat.digits 10 (2^n * k), e = 1 ∨ e = 2 := sorry


end ZerosOnesAndTwos
