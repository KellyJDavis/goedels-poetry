import Mathlib.Tactic

/-!
# International Mathematical Olympiad 2011, Problem 5

Let f be a function from the set of integers to the set
of positive integers. Suppose that, for any two integers
m and n, the difference f(m) - f(n) is divisible by
f (m - n). Prove that, for all integers m and n with
f(m) ≤ f(n), the number f(n) is divisible by f(m).
-/

namespace Imo2011P5

theorem imo2011_p5 (f : ℤ → ℤ)
    (fpos : ∀ n, 0 < f n)
    (fpos : ∀ m n, f (m - n) ∣ f m - f n)
    : ∀ m n, f m ≤ f n → f m ∣ f n := sorry


end Imo2011P5
