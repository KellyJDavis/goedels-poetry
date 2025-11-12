
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1960, Problem 1

Determine all three-digit numbers N having the property that N is divisible by 11, and
N/11 is equal to the sum of the squares of the digits of N.
-/
def sumOfSquares (L : List ℕ) : ℕ :=
  (L.map fun x => x * x).sum

def ProblemPredicate (n : ℕ) : Prop :=
  (Nat.digits 10 n).length = 3 ∧ 11 ∣ n ∧ n / 11 = sumOfSquares (Nat.digits 10 n)

abbrev SolutionPredicate (n : ℕ) : Prop :=
  n = 550 ∨ n = 803

theorem imo1960_p1 (n : ℕ) : ProblemPredicate n ↔ SolutionPredicate n := by sorry
