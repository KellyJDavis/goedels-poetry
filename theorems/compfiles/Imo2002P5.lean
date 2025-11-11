import Mathlib.Tactic

/-!
# International Mathematical Olympiad 2002, Problem 5

Determine all functions f : ℝ → ℝ such that

  (f(x) + f(z))(f(y) + f(t)) = f(xy - zt) + f(xt + yz)

for all real numbers x,y,z,t.
-/

namespace Imo2002P5

/- determine -/ abbrev SolutionSet : Set (ℝ → ℝ) := sorry

theorem imo2002_p5 (f : ℝ → ℝ) :
    f ∈ SolutionSet ↔
    ∀ x y z t, (f x + f z) * (f y + f t) =
               f (x * y - z * t) + f (x * t + y * z) := sorry



end Imo2002P5
