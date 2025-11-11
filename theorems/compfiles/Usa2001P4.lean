import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Inversion.Basic
import Mathlib.Geometry.Euclidean.Sphere.Basic
import Mathlib.Geometry.Euclidean.Triangle

/-!
# USA Mathematical Olympiad 2001, Problem 4

Let ABC be a triangle and P be any point such that PA, PB, PC
are the sides of an obtuse triangle, with PA the longest side.
Prove that ∠BAC is acute.
-/

namespace Usa2001P4

open scoped EuclideanGeometry

theorem usa2001_p4
    (A B C P X Y Z : EuclideanSpace ℝ (Fin 2))
    (hABC : AffineIndependent ℝ ![A, B, C])
    (hXYZ : AffineIndependent ℝ ![X, Y, Z])
    (hPA : dist X Y = dist P A)
    (hPB : dist Y Z = dist P B)
    (hPC : dist Z X = dist P C)
    (hObtuse : Real.pi / 2 < ∠ X Z Y)
    : ∠ B A C < Real.pi / 2 := sorry


end Usa2001P4
