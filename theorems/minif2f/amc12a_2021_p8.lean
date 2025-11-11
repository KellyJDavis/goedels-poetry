import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- A sequence of numbers is defined by $D_0=0,D_1=0,D_2=1$ and $D_n=D_{n-1}+D_{n-3}$ for $n\ge 3$. What are the parities (evenness or oddness) of the triple of numbers $(D_{2021},D_{2022},D_{2023})$, where $E$ denotes even and $O$ denotes odd?

$\textbf{(A) }(O,E,O) \qquad \textbf{(B) }(E,E,O) \qquad \textbf{(C) }(E,O,E) \qquad \textbf{(D) }(O,O,E) \qquad \textbf{(E) }(O,O,O)$ Show that it is \textbf{(C) }(E,O,E).-/
theorem amc12a_2021_p8 (d : ℕ → ℕ) (h₀ : d 0 = 0) (h₁ : d 1 = 0) (h₂ : d 2 = 1)
    (h₃ : ∀ n ≥ 3, d n = d (n - 1) + d (n - 3)) : Even (d 2021) ∧ Odd (d 2022) ∧ Even (d 2023) := by sorry
