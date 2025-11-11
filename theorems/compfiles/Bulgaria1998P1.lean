import Mathlib.Tactic.NormNum
import Mathlib.Tactic.IntervalCases

/-!
# Bulgarian Mathematical Olympiad 1998, Problem 1

We will be considering colorings in 2 colors of n (distinct) points
A₁, A₂, ..., Aₙ. Call such a coloring "good" if there exist three points
Aᵢ, Aⱼ, A₂ⱼ₋ᵢ, 1 ≤ i < 2j - i ≤ n, which are colored the same color.

Find the least natural number n (n ≥ 3) such that all colorings
of n points are good.
-/

namespace Bulgaria1998P1

abbrev coloring_is_good {m : ℕ} (color : Set.Icc 1 m → Fin 2) : Prop :=
  ∃ i j : Set.Icc 1 m,
    i < j ∧
    ∃ h3 : 2 * j.val - i ∈ Set.Icc 1 m,
    color i = color j ∧ color i = color ⟨2 * j - i, h3⟩

abbrev all_colorings_are_good (m : ℕ) : Prop :=
  3 ≤ m ∧ ∀ color : Set.Icc 1 m → Fin 2, coloring_is_good color

/- determine -/ abbrev solution_value : ℕ := sorry

theorem bulgaria1998_p1 : IsLeast { m | all_colorings_are_good m } solution_value := sorry


end Bulgaria1998P1
