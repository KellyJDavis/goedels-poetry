
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1962, Problem 1

Find the smallest natural number $n$ which has the following properties:

(a) Its decimal representation has 6 as the last digit.

(b) If the last digit 6 is erased and placed in front of the remaining digits,
the resulting number is four times as large as the original number $n$.
-/
def ProblemPredicate (n : ℕ) : Prop :=
  (digits 10 n).headI = 6 ∧ ofDigits 10 ((digits 10 n).tail.concat 6) = 4 * n

abbrev solution : ℕ := 153846

theorem imo1962_p1 : IsLeast {n | ProblemPredicate n} solution := by sorry
