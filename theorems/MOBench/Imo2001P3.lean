
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2001, Problem 3

Twenty-one girls and twenty-one boys took part in a mathematical competition.
It turned out that each contestant solved at most six problems, and for each
pair of a girl and a boy, there was at most one problem solved by both the
girl and the boy. Show that there was a problem solved by at least three
girls and at least three boys.
-/
/- special open -/ open Finset






/-- A problem is easy for a cohort (boys or girls) if at least three
    of its members solved it. -/

def Easy {α : Type} [Fintype α] (F : α → Finset ℕ) (p : ℕ) : Prop :=
  3 ≤ Finset.card (filter (λ i => p ∈ F i) (univ : Finset α))

theorem imo2001_p3
    {Girl Boy : Type}
    [Fintype Girl] [Fintype Boy] [DecidableEq Girl] [DecidableEq Boy]
    {G : Girl → Finset ℕ} {B : Boy → Finset ℕ} -- solved problems
    (hcard_girl : 21 = Fintype.card Girl)
    (hcard_boy : 21 = Fintype.card Boy)
    (G_le_6 : ∀ i, Finset.card (G i) ≤ 6) -- Every girl solved at most six problems.
    (B_le_6 : ∀ j, Finset.card (B j) ≤ 6) -- Every boy solved at most six problems.
    (G_inter_B : ∀ i j, ¬Disjoint (G i) (B j)) :
    ∃ p, Easy G p ∧ Easy B p := by sorry
