import Mathlib.Algebra.Ring.Equiv
import Mathlib.Data.Real.Basic
import Mathlib.Topology.Order
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Topology.Algebra.Ring.Basic
import Mathlib.Topology.Bases
import Mathlib.Tactic.Linarith

/-!
Suppose f : ℝ -> ℝ is continuous in both the upper topology (where
the basic open sets are half-open intervals (a, b]) and lower topology
(where the basic open sets are half-open intervals [a,b)).
Then f is continuous in the usual topology (where the basic open sets are
open intervals (a,b)) and also monotone nondecreasing.
-/

namespace UpperLowerContinuous

def upper_intervals : Set (Set ℝ) := {s : Set ℝ | ∃ a b : ℝ, Set.Ioc a b = s}
def lower_intervals : Set (Set ℝ) := {s : Set ℝ | ∃ a b : ℝ, Set.Ico a b = s}
def open_intervals : Set (Set ℝ) := {s : Set ℝ | ∃ a b : ℝ, Set.Ioo a b = s}

/-- Generate the toplogy on ℝ by intervals of the form (a, b]. -/
def tᵤ : TopologicalSpace ℝ := TopologicalSpace.generateFrom upper_intervals

/-- Generate the toplogy on ℝ by intervals of the form [a, b). -/
def tₗ : TopologicalSpace ℝ := TopologicalSpace.generateFrom lower_intervals

/-- This should be equivalent to the default instance
for `TopologicalSpace ℝ`, which goes through `UniformSpace`, but for
now I don't want to bother with proving that equivalence.
-/
def tₛ : TopologicalSpace ℝ := TopologicalSpace.generateFrom open_intervals

-- activate the Continuous[t1, t2] notation
open Topology

theorem properties_of_upper_lower_continuous
    (f : ℝ → ℝ)
    (huc : Continuous[tᵤ, tᵤ] f)
    (hlc : Continuous[tₗ, tₗ] f)
    : Continuous[tₛ, tₛ] f ∧ Monotone f := sorry


end UpperLowerContinuous
