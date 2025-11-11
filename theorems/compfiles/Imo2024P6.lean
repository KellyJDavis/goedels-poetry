import Mathlib.Data.Rat.Floor
import Mathlib.SetTheory.Cardinal.Basic


/-!
# International Mathematical Olympiad 2024, Problem 6

A function `f: ℚ → ℚ` is called *aquaesulian* if the following
property holds: for every `x, y ∈ ℚ`,
`f(x + f(y)) = f(x) + y` or `f(f(x) + y) = x + f(y)`.

Show that there exists an integer `c` such that for any aquaesulian function `f`
there are at most `c` different rational numbers of the form `f(r)+f(-r)` for
some rational number `r`, and find the smallest possible value of `c`.
-/

namespace Imo2024P6

def Aquaesulian (f : ℚ → ℚ) : Prop :=
  ∀ x y, f (x + f y) = f x + y ∨ f (f x + y) = x + f y

open scoped Cardinal

/- determine -/ abbrev solution : ℕ := sorry

theorem imo2024_p6 :
    (∀ f, Aquaesulian f → #(Set.range (fun x ↦ f x + f (-x))) ≤ solution) ∧
    ∀ c : ℕ,
      (∀ f, Aquaesulian f → #(Set.range (fun x ↦ f x + f (-x))) ≤ c) →
        solution ≤ c := sorry

end Imo2024P6
