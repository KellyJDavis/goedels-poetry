
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1993, Problem 5

Does there exist a function f : ℕ → ℕ such that
  i) f(1) = 2
  ii) f(f(n)) = f(n) + n for all n ∈ ℕ
  iii) f(n + 1) > f(n) for all n ∈ ℕ?
-/
abbrev DoesExist : Bool := True

abbrev Good (f : ℕ → ℕ) : Prop := f 1 = 2 ∧ ∀ n, f (f n) = f n + n ∧ ∀ n, f n < f (n + 1)

theorem imo1993_p5 :
    if DoesExist then ∃ f, Good f else ¬∃ f, Good f := by sorry
