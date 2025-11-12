
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2020, Problem 5

A deck of n > 1 cards is given. A positive integer is written on
each card. The deck has the property that the arithmetic mean of
the numbers on each pair of cards is also the geometric mean of
the numbers on some collection of one or more cards.

For which n does it follow that the numbers on the cards are all equal?
-/
abbrev SolutionSet : Set ℕ := {n : ℕ | n > 1}

noncomputable def geometric_mean {α : Type} (f : α → ℕ+) (s : Finset α) : ℝ :=
  (∏ i ∈ s, (f i : ℝ))^((1:ℝ)/s.card)

theorem imo2020_p5 (n : ℕ) :
    n ∈ SolutionSet ↔
    (1 < n ∧
     (∀ α : Type, [Fintype α] → Fintype.card α = n →
         ∀ f : α → ℕ+,
           (Pairwise fun a b ↦ ∃ s : Finset α,
              s.Nonempty ∧ geometric_mean f s = (((f a):ℝ) + f b) / 2)
           → ∃ y, ∀ a, f a = y )) := by sorry
