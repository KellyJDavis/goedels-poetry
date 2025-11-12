
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# USA Mathematical Olympiad 2011, Problem 4

For any integer n ≥ 2, define P(n) to be the proposition:

  P(n) ≡  2^(2^n) % (2^n - 1) is a power of 4

Either prove that P(n) is always true, or find a counterexample.
-/
abbrev P (n : ℕ) : Prop := ∃ k, 4^k = 2^(2^n) % (2^n - 1)

inductive SolutionData where
| AlwaysTrue : SolutionData
| Counterexample : ℕ → SolutionData

abbrev solution_data : SolutionData := SolutionData.Counterexample 25

theorem usa2011_p4 :
    match solution_data with
    | .AlwaysTrue => ∀ n, 2 ≤ n → P n
    | .Counterexample m => 2 ≤ m ∧ ¬ P m := by sorry
