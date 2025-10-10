# Answer: Are Hypothesis Statements Also Properly Typed?

## Short Answer

**YES!** Hypothesis statements from earlier `have` statements are also properly typed when they appear as dependencies in extracted subgoals.

## How It Works

When you extract a subgoal like `hOA_parallel_CA_or` using `AST.get_named_subgoal_code()` with sorries, the implementation:

### 1. Identifies Dependencies

The code finds all named declarations (variables and hypotheses) that are referenced in the target subgoal using `__find_dependencies()`.

### 2. Extracts Type Information from Goal Context

The goal string in sorries contains type information for **both variables AND hypotheses**:

```
O A C B D : ℂ                                  # Variable types
hd₁ : ¬B = D                                    # Hypothesis type
hd₂ : ¬C = D                                    # Hypothesis type
hnorm : ‖O - A‖ * ‖D - B‖ = ‖A - C‖ * ‖B - O‖  # Hypothesis type
hOAcol : ((O - A) / (C - A)).im = 0            # Hypothesis type
hOBcol : ((O - B) / (D - B)).im = 0            # Hypothesis type
⊢ goal_expression                               # The actual goal
```

### 3. Creates Typed Binders

For each dependency (whether variable or hypothesis), the code now:
1. **First checks** if type information is available from the goal context (sorries)
2. **Uses that type** to create a properly-typed binder
3. Falls back to AST extraction only if goal context is not available

The key code (recently fixed):

```python
for d in sorted(deps):
    # Prioritize goal context types (from sorries) as they're more specific and complete
    if d in goal_var_types:
        binder = __make_binder_from_type_string(d, goal_var_types[d])
    else:
        # Fall back to AST extraction if no goal context available
        dep_node = name_map.get(d)
        dep_type_ast = __extract_type_ast(dep_node) if dep_node is not None else None
        binder = __make_binder(d, dep_type_ast)
    binders.append(binder)
```

## Example Output

For the subgoal `hOA_parallel_CA_or` in your code, assuming it depends on:
- Variables: O, A, C (all of type ℂ)
- Hypothesis: hOAcol

The generated code would be:

```lean
lemma hOA_parallel_CA_or
  (O : ℂ)
  (A : ℂ)
  (C : ℂ)
  (hOAcol : ((O - A) / (C - A)).im = 0)
  : C = A ∨ ∃ r : ℝ, O - A = r • (C - A) := by sorry
```

Notice that:
- ✅ Variables have their types: `(O : ℂ)`, `(A : ℂ)`, `(C : ℂ)`
- ✅ Hypothesis also has its full type: `(hOAcol : ((O - A) / (C - A)).im = 0)`

## What Gets Typed

The extracted subgoal code includes **complete type information** for:

1. **Variables** - Like `O A C B D : ℂ`
   - Example: `(O : ℂ) (A : ℂ) (C : ℂ)`

2. **Theorem Parameters** - Like `hd₁ : ¬B = D`
   - Example: `(hd₁ : ¬B = D)`

3. **Earlier Hypotheses** - Like `hnorm : ‖O - A‖ * ‖D - B‖ = ...`
   - Example: `(hnorm : ‖O - A‖ * ‖D - B‖ = ‖A - C‖ * ‖B - O‖)`

4. **Earlier Have Statements** - Like `hOAcol : ((O - A) / (C - A)).im = 0`
   - Example: `(hOAcol : ((O - A) / (C - A)).im = 0)`

## Why This Works

The sorries list from the Kimina check response contains a `goal` field that shows the **complete proof state** at the point of each sorry. This includes:

- All variables in scope with their types
- All hypotheses (from theorem parameters) with their types
- All earlier have statements with their types
- The current goal

By parsing this rich information, we can reconstruct complete type signatures for any dependencies.

## Implementation Note

The recent fix **prioritizes** goal context types over AST-extracted types because:

1. **More Specific**: Goal context has the exact, user-written type expressions
2. **More Complete**: It includes complex types that might be hard to extract from AST
3. **More Accurate**: It reflects the actual Lean type system's understanding

## Verification

You can verify this by:

1. Creating an AST with sorries from a check response
2. Extracting a subgoal that depends on earlier hypotheses
3. Checking that the generated code includes both variable and hypothesis types

```python
from goedels_poetry.parsers.ast import AST

# ast_dict from ast_code response
# sorries from check response
ast = AST(ast_dict, sorries)

code = ast.get_named_subgoal_code("hOA_parallel_CA_or")
print(code)
# Will include types for both variables AND hypotheses
```

## Summary

**Yes, hypothesis statements are properly typed!**

The implementation extracts type information from the goal context for:
- ✅ Variables (O, A, C, B, D, ...)
- ✅ Theorem parameters (hd₁, hd₂, ...)
- ✅ Intro'd hypotheses (hnorm, hOAcol, ...)
- ✅ Earlier have statements

All dependencies that appear in the goal context will have their complete type information included in the generated subgoal code.
