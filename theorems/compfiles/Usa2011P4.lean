import Mathlib.Tactic

/-!
# USA Mathematical Olympiad 2011, Problem 4

For any integer n ≥ 2, define P(n) to be the proposition:

  P(n) ≡  2^(2^n) % (2^n - 1) is a power of 4

Either prove that P(n) is always true, or find a counterexample.
-/

namespace Usa2011P4

abbrev P (n : ℕ) : Prop := ∃ k, 4^k = 2^(2^n) % (2^n - 1)

inductive SolutionData where
| AlwaysTrue : SolutionData
| Counterexample : ℕ → SolutionData

/- determine -/ abbrev solution_data : SolutionData := sorry

theorem usa2011_p4 :
    match solution_data with
    | .AlwaysTrue => ∀ n, 2 ≤ n → P n
    | .Counterexample m => 2 ≤ m ∧ ¬ P m := sorry


end Usa2011P4
