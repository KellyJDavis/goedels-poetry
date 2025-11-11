import Mathlib.Algebra.BigOperators.Group.Finset
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Data.Finset.Card
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# USA Mathematical Olympiad 1998, Problem 5

Prove that for each n ≥ 2, there is a set S of n integers such that
(a-b)² divides ab for every distinct a,b ∈ S.
-/

namespace Usa1998P5

theorem usa1998_p5 (n : ℕ) (_hn : 2 ≤ n) :
    ∃ S : Finset ℤ,
       S.card = n ∧
       ∀ a ∈ S, ∀ b ∈ S, a ≠ b → (a - b)^2 ∣ a * b := sorry


end Usa1998P5
