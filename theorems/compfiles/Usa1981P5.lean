import Mathlib.Tactic

/-!
# USA Mathematical Olympiad 1981, Problem 5

Show that for any positive real number x and any nonnegative
integer n,

    ∑ₖ (⌊kx⌋/k) ≤ ⌊nx⌋

where the sum goes from k = 1 to k = n, inclusive.
-/

namespace Usa1981P5

theorem usa1981_p5 (x : ℝ) (n : ℕ) :
    ∑ k ∈ Finset.Icc 1 n, ((⌊k * x⌋:ℝ)/k) ≤ ⌊n * x⌋ := sorry


end Usa1981P5
