
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1994, Problem 5

Let S be the set of all real numbers greater than -1.
Find all functions f : S→S such that f(x + f(y) + xf(y)) = y + f(x) + yf(x)
for all x and y, and f(x)/x is strictly increasing on each of the
intervals -1 < x < 0 and 0 < x.
-/
def S := { x : ℝ // -1 < x }

abbrev op (f : S → S) (a b : S) : S :=
  ⟨a.val + (f b).val + a.val * (f b).val, by nlinarith [a.property, (f b).property]⟩

axiom sol_prop {a : ℝ} (ha : -1 < a) : -1 < -a / (1 + a)

abbrev solution_set : Set (S → S) := { fun x ↦ ⟨-x.val / (1 + x.val), sol_prop x.property⟩ }

theorem imo1994_p5 (f : S → S) :
    f ∈ solution_set ↔
    ((∀ x y : S, f (op f x y) = op f y x) ∧
     (∀ x y : S, x.val ∈ Set.Ioo (-1) 0 → y.val ∈ Set.Ioo (-1) 0 →
        x.val < y.val → (f x).val / x.val < (f y).val / y.val) ∧
     (∀ x y : S, 0 < x.val → 0 < y.val →
        x.val < y.val → (f x).val / x.val < (f y).val / y.val)) := by sorry
