
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2020, Problem 3

There are 4n pebbles of weights 1,2,3,...,4n. Each pebble is colored
in one of n colors and there are four pebbles of each color. Show
that we can arrange the pebbles into two piles such that the total
weights of both piles are the same, and each pile contains two
pebbles of each color.
-/
/- special open -/ open Finset






theorem imo2020_p3 {n : ℕ} {c : Fin (4 * n) → Fin n} (h : ∀ i, Finset.card (filter (λ j => c j = i) univ) = 4) :
    ∃ S : Finset (Fin (4 * n)), ∑ i ∈ S, ((i : ℕ) + 1) = ∑ i ∈ Sᶜ, ((i : ℕ) + 1) ∧
      ∀ i, Finset.card (filter (λ j => c j = i) S) = 2 := by sorry
