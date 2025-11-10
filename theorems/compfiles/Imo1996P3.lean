import Mathlib.Tactic

/-!
# International Mathematical Olympiad 1996, Problem 3

Let S denote the set of nonnegative integers. Find
all functions f from S to itself such that

 f(m + f(n)) = f(f(m)) + f(n)

for all m,n in S.
-/

namespace Imo1996P3

/- determine -/ abbrev SolutionSet : Set (ℕ → ℕ) := sorry

theorem imo1996_p3 (f : ℕ → ℕ) :
    f ∈ SolutionSet ↔ ∀ m n, f (m + f n) = f (f m) + f n := sorry


end Imo1996P3
