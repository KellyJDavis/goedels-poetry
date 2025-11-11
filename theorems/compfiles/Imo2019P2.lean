import Mathlib.Geometry.Euclidean.Angle.Sphere
import Mathlib.Geometry.Euclidean.Sphere.SecondInter

/-!
# International Mathematical Olympiad 2019, Problem 2

In triangle `ABC`, point `A₁` lies on side `BC` and point `B₁` lies on side `AC`. Let `P` and
`Q` be points on segments `AA₁` and `BB₁`, respectively, such that `PQ` is parallel to `AB`.
Let `P₁` be a point on line `PB₁`, such that `B₁` lies strictly between `P` and `P₁`, and
`∠PP₁C = ∠BAC`. Similarly, let `Q₁` be a point on line `QA₁`, such that `A₁` lies strictly
between `Q` and `Q₁`, and `∠CQ₁Q = ∠CBA`.

Prove that points `P`, `Q`, `P₁`, and `Q₁` are concyclic.
-/

open Affine Affine.Simplex EuclideanGeometry FiniteDimensional Module

open scoped Affine EuclideanGeometry Real

attribute [local instance] FiniteDimensional.of_fact_finrank_eq_two

variable (V : Type*) (Pt : Type*)

variable [NormedAddCommGroup V] [InnerProductSpace ℝ V] [MetricSpace Pt]

variable [NormedAddTorsor V Pt]

namespace Imo2019P2

theorem imo2019_p2 [Fact (finrank ℝ V = 2)] (A B C A₁ B₁ P Q P₁ Q₁ : Pt)
    (affine_independent_ABC : AffineIndependent ℝ ![A, B, C]) (wbtw_B_A₁_C : Wbtw ℝ B A₁ C)
    (wbtw_A_B₁_C : Wbtw ℝ A B₁ C) (wbtw_A_P_A₁ : Wbtw ℝ A P A₁) (wbtw_B_Q_B₁ : Wbtw ℝ B Q B₁)
    (PQ_parallel_AB : line[ℝ, P, Q] ∥ line[ℝ, A, B]) (P_ne_Q : P ≠ Q)
    (sbtw_P_B₁_P₁ : Sbtw ℝ P B₁ P₁) (angle_PP₁C_eq_angle_BAC : ∠ P P₁ C = ∠ B A C)
    (C_ne_P₁ : C ≠ P₁) (sbtw_Q_A₁_Q₁ : Sbtw ℝ Q A₁ Q₁)
    (angle_CQ₁Q_eq_angle_CBA : ∠ C Q₁ Q = ∠ C B A) (C_ne_Q₁ : C ≠ Q₁) :
    Concyclic ({P, Q, P₁, Q₁} : Set Pt) := sorry


end Imo2019P2
