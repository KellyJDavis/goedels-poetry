import Mathlib.Tactic

/-!
# USA Mathematical Olympiad 1989, Problem 5

Let u and v be real numbers such that

(u + u² + u³ + ⋯ + u⁸) + 10u⁹ = (v + v² + v³ + ⋯ + v¹⁰) + 10v¹¹ = 8.

Determine, with proof, which of the two numbers, u or v, is larger.
-/

namespace Usa1989P5

/- determine -/ abbrev u_is_larger : Bool := sorry

theorem usa1989_p5
    (u v : ℝ)
    (hu : (∑ i in Finset.range 8, u^(i + 1)) + 10 * u^9 = 8)
    (hv : (∑ i in Finset.range 10, v^(i + 1)) + 10 * v^11 = 8) :
    if u_is_larger then v < u else u < v := sorry


end Usa1989P5
