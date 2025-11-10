import Mathlib.Data.Nat.Digits

/-!
# International Mathematical Olympiad 1960, Problem 1

Determine all three-digit numbers N having the property that N is divisible by 11, and
N/11 is equal to the sum of the squares of the digits of N.
-/

open Nat

namespace Imo1960P1

def sumOfSquares (L : List ℕ) : ℕ :=
  (L.map fun x => x * x).sum

def ProblemPredicate (n : ℕ) : Prop :=
  (Nat.digits 10 n).length = 3 ∧ 11 ∣ n ∧ n / 11 = sumOfSquares (Nat.digits 10 n)

/- determine -/ abbrev SolutionPredicate (n : ℕ) : Prop := sorry

theorem imo1960_p1 (n : ℕ) : ProblemPredicate n ↔ SolutionPredicate n := sorry



end Imo1960P1
