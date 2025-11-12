
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2012, Problem 4

Determine all functions f : ℤ → ℤ such that for all integers a,b,c with a + b + c = 0,
the following equality holds:
  f(a)² + f(b)² + f(c)² = 2f(a)f(b) + 2f(b)f(c) + 2f(c)f(a).
-/
def odd_const : Set (ℤ → ℤ) := fun f =>
  ∃ c : ℤ, ∀ x : ℤ,
    (Odd x → f x = c) ∧ (Even x → f x = 0)

def mod4_cycle : Set (ℤ → ℤ) := fun f =>
  ∃ c : ℤ, ∀ x : ℤ, f x =
  match x % 4 with
    | 0 => 0
    | 2 => 4 * c
    | _ => c

def square_set : Set (ℤ → ℤ) := fun f =>
  ∃ c : ℤ, ∀ x : ℤ, f x = x ^ 2 * c

abbrev solution_set : Set (ℤ → ℤ) := odd_const ∪ mod4_cycle ∪ square_set

theorem imo2012_p4 (f : ℤ → ℤ) :
    f ∈ solution_set ↔
    ∀ a b c : ℤ, a + b + c = 0 →
      (f a)^2 + (f b)^2 + (f c)^2 =
        2 * f a * f b + 2 * f b * f c + 2 * f c * f a := by sorry
