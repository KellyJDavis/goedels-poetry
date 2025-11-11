import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- How many positive integers $n$ satisfy $\dfrac{n+1000}{70} = \lfloor \sqrt{n} \rfloor?$(Recall that $\lfloor x\rfloor$ is the greatest integer not exceeding $x$.)

$\textbf{(A) } 2 \qquad\textbf{(B) } 4 \qquad\textbf{(C) } 6 \qquad\textbf{(D) } 30 \qquad\textbf{(E) } 32$ Show that it is \textbf{(C) }6.-/
theorem amc12b_2020_p21 (S : Finset ℕ)
    (h₀ : ∀ n : ℕ, n ∈ S ↔ 0 < n ∧ (↑n + (1000 : ℝ)) / 70 = Int.floor (Real.sqrt n)) : S.card = 6 := by sorry
