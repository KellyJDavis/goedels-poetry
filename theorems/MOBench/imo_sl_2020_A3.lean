
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2020 A3

Let $a, b, c, d$ be positive real numbers such that $(a + c)(b + d) = ac + bd$.
Find the smallest possible value of
$$ \frac{a}{b} + \frac{b}{c} + \frac{c}{d} + \frac{d}{a}. $$
-/
variable {F : Type*} [LinearOrderedField F]

def IsGood (a b c d : F) : Prop :=
  (a + c) * (b + d) = a * c + b * d

def targetVal (a b c d : F) : F :=
  a / b + b / c + c / d + d / a

theorem imo_sl_2020_A3 :
  (∀ a b c d : F, 0 < a → 0 < b → 0 < c → 0 < d → IsGood a b c d → (8 : F) ≤ targetVal a b c d) ∧
  (∃ a b c d : F, 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ IsGood a b c d ∧ targetVal a b c d = (8 : F)) :=
  by sorry
