import Mathlib.Tactic
import Mathlib.Data.Set.Card

/-!
# International Mathematical Olympiad 2008, Problem 5

Let n and k be positive integers with k ≥ n and k - n an even number.
There are 2n lamps labelled 1, 2, ..., 2n each of which can be
either on or off. Initially all the lamps are off. We consider
sequences of steps: at each step one of the lamps is switched (from
on to off or from off to on). Let N be the number of such sequences
consisting of k steps and resulting in the state where lamps 1 through
n are all on, and lamps n + 1 through 2n are all off. Let M be the
number of such sequences consisting of k steps, resulting in the state
where lamps 1 through n are all on, and lamps n + 1 through 2n are all off,
but where none of the lamps n + 1 through 2n is ever switched on.

Determine N/M.
-/

namespace Imo2008P5

abbrev Sequence (n k : ℕ) := Fin k → Fin (2 * n)

abbrev NSequence (n k : ℕ) (f : Sequence n k) : Prop :=
  (∀ i < n, Odd (Nat.card { j | f j = i })) ∧
  (∀ i, n ≤ i → i < 2 * n → Even (Nat.card { j | f j = i }))

abbrev MSequence (n k : ℕ) (f : Sequence n k) : Prop :=
  NSequence n k f ∧
  (∀ i : Fin (2 * n), n ≤ i → ∀ j : Fin k, f j ≠ i)

/- determine -/ abbrev solution (n k : ℕ) : ℚ := sorry

theorem imo2008_p5 (n k : ℕ) (hn : 0 < n)
    (hnk : n ≤ k) (he : Even (k - n))
    : Set.ncard (MSequence n k) * solution n k = Set.ncard (NSequence n k) := sorry


end Imo2008P5
