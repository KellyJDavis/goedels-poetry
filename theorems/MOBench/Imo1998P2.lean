
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1998, Problem 2
In a competition, there are `a` contestants and `b` judges, where `b ≥ 3` is an odd integer. Each
judge rates each contestant as either "pass" or "fail". Suppose `k` is a number such that, for any
two judges, their ratings coincide for at most `k` contestants.

Prove that `k / a ≥ (b - 1) / (2b)`.
-/
/- special open -/ open Classical





variable {C J : Type*} (r : C → J → Prop)


noncomputable section

/-- An ordered pair of judges. -/
abbrev JudgePair (J : Type*) :=
  J × J

/-- The first judge from an ordered pair of judges. -/
abbrev JudgePair.judge₁ : JudgePair J → J :=
  Prod.fst

/-- The second judge from an ordered pair of judges. -/
abbrev JudgePair.judge₂ : JudgePair J → J :=
  Prod.snd

/-- The proposition that the judges in an ordered pair are distinct. -/
abbrev JudgePair.Distinct (p : JudgePair J) :=
  p.judge₁ ≠ p.judge₂

/-- The proposition that the judges in an ordered pair agree about a contestant's rating. -/
abbrev JudgePair.Agree (p : JudgePair J) (c : C) :=
  r c p.judge₁ ↔ r c p.judge₂

/-- The set of contestants on which two judges agree. -/
def agreedContestants [Fintype C] (p : JudgePair J) : Finset C :=
  Finset.univ.filter fun c => p.Agree r c


theorem imo1998_p2 [Fintype J] [Fintype C] (a b k : ℕ) (hC : Fintype.card C = a)
    (hJ : Fintype.card J = b) (ha : 0 < a) (hb : Odd b)
    (hk : ∀ p : JudgePair J, p.Distinct → (agreedContestants r p).card ≤ k) :
    (b - 1 : ℚ) / (2 * b) ≤ k / a := by sorry
