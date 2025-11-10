import Mathlib.Data.Nat.Digits

/-!
# International Mathematical Olympiad 1962, Problem 1

Find the smallest natural number $n$ which has the following properties:

(a) Its decimal representation has 6 as the last digit.

(b) If the last digit 6 is erased and placed in front of the remaining digits,
the resulting number is four times as large as the original number $n$.
-/


namespace Imo1962P1

open Nat

def ProblemPredicate (n : ℕ) : Prop :=
  (digits 10 n).headI = 6 ∧ ofDigits 10 ((digits 10 n).tail.concat 6) = 4 * n

/- determine -/ abbrev solution : ℕ := sorry

theorem imo1962_p1 : IsLeast {n | ProblemPredicate n} solution := sorry



end Imo1962P1
