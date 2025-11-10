import Mathlib.Algebra.Order.Positive.Field
import Mathlib.Tactic

/-!
# International Mathematical Olympiad 2008, Problem 4

Determine all functions f from the positive reals to the positive reals
such that

   (f(w)² + f(x)²) / (f(y)² + f(z)²) = (w² + x²) / (y² + z²)

for all positive real numbers w,x,y,z satisfying xw = yz.
-/

namespace Imo2008P4

abbrev PosReal : Type := { x : ℝ // 0 < x }

/- determine -/ abbrev solution_set : Set (PosReal → PosReal) := sorry

theorem imo2008_p4 (f : PosReal → PosReal) :
    f ∈ solution_set ↔
      ∀ w x y z, w * x = y * z →
       ((f w)^2 + (f x)^2) * (y^2 + z^2) = (w^2 + x^2) * (f (y^2) + f (z^2)) := sorry


end Imo2008P4
