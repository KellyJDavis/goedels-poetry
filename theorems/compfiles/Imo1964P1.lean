import Mathlib.Data.Nat.ModEq
import Mathlib.Tactic

/-!
# International Mathematical Olympiad 1964, Problem 1

(a) Find all natural numbers n for which 2ⁿ - 1 is divisible by 7.
(b) Prove that there is no positive integer n for which 2ⁿ + 1 is divisible by 7.
-/

namespace Imo1964P1

/- determine -/ abbrev solution_set : Set ℕ := sorry

theorem imo_1964_p1a (n : ℕ) : n ∈ solution_set ↔ 2^n ≡ 1 [MOD 7] := sorry

theorem imo_1964_p1b (n : ℕ) : ¬ 7 ∣ (2^n + 1) := sorry

end Imo1964P1
