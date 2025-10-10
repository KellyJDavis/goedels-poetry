# AST Parser Enhancements: Type Information Extraction from Sorries

## Summary

Enhanced the AST parser to extract and utilize type information from the `sorries` list provided by the Kimina server's check response. This allows generated subgoal lemmas to include complete type declarations for all referenced variables.

## Problem

Previously, when extracting named subgoals using `AST.get_named_subgoal_code()`, the generated Lean code did not include type information for variables. For example:

**Before:**
```lean
lemma hOA_parallel_CA_or : C = A ∨ ∃ r, O - A = r • (C - A) := by sorry
```

This code is incomplete because it doesn't declare the types of variables like `C`, `A`, `O`, etc., even though this information was available in the `sorries` list from the check response.

## Solution

Modified the AST class and parsing utilities to:
1. Accept an optional `sorries` parameter containing goal context from check responses
2. Parse the `goal` field in sorries to extract variable type declarations
3. Add these type declarations as binders when generating subgoal code

**After:**
```lean
lemma hOA_parallel_CA_or (O : ℂ) (A : ℂ) (C : ℂ) : C = A ∨ ∃ r, O - A = r • (C - A) := by sorry
```

## Changes Made

### 1. `goedels_poetry/parsers/ast.py`

#### Modified `AST.__init__`
- Added optional `sorries` parameter
- Stores sorries list for use in subgoal extraction

```python
def __init__(self, ast: dict[str, Any], sorries: list[dict[str, Any]] | None = None):
    self._ast: dict[str, Any] = ast
    self._sorries: list[dict[str, Any]] = sorries or []
```

#### Modified `AST.get_named_subgoal_code`
- Passes sorries to the rewriter function

```python
def get_named_subgoal_code(self, subgoal_name: str) -> str:
    rewritten_subgoal_ast = _get_named_subgoal_rewritten_ast(self._ast, subgoal_name, self._sorries)
    return str(_ast_to_code(rewritten_subgoal_ast))
```

### 2. `goedels_poetry/parsers/util.py`

#### Added `__parse_goal_context(goal: str) -> dict[str, str]`
Parses the goal string from sorries to extract variable type declarations.

Example input:
```
O A C B D : ℂ
hd₁ : ¬B = D
hd₂ : ¬C = D
⊢ goal_expression
```

Returns: `{"O": "ℂ", "A": "ℂ", "C": "ℂ", "B": "ℂ", "D": "ℂ", "hd₁": "¬B = D", "hd₂": "¬C = D"}`

#### Added `__make_binder_from_type_string(name: str, type_str: str) -> dict`
Creates an AST binder node from a variable name and type string.

#### Added `__is_referenced_in(subtree: Node, name: str) -> bool`
Checks if a variable is referenced in an AST subtree (excluding binding occurrences).

#### Modified `_get_named_subgoal_rewritten_ast`
- Added `sorries` parameter
- Parses goal context from sorries to extract type information
- Uses goal context to create properly-typed binders for variables
- Adds binders for all referenced variables with known types

## Usage

### Basic Usage (Backward Compatible)

```python
from goedels_poetry.parsers.ast import AST

# Old way - still works, but won't have type information
ast = AST(ast_dict)
code = ast.get_named_subgoal_code("subgoal_name")
```

### Enhanced Usage (With Type Information)

```python
from goedels_poetry.parsers.ast import AST

# New way - includes type information from sorries
ast = AST(ast_dict, sorries)
code = ast.get_named_subgoal_code("subgoal_name")
```

### Complete Example

```python
from kimina_client import KiminaClient
from goedels_poetry.parsers.ast import AST

# Get AST and check response from Kimina server
kimina_client = KiminaClient(api_url="http://0.0.0.0:8000")

# Get the AST
ast_response = kimina_client.ast_code(lean_code)
ast_dict = ast_response.results[0].ast

# Get the check response with sorries
check_response = kimina_client.check(lean_code)
sorries = check_response.results[0].response.get("sorries", [])

# Create AST with sorries
ast = AST(ast_dict, sorries)

# Extract subgoal with complete type information
subgoal_code = ast.get_named_subgoal_code("hOA_parallel_CA_or")
```

## Testing

Added comprehensive tests in `tests/test_parsers_ast.py`:
- `test_ast_init_with_sorries()` - Verifies AST initialization with sorries
- `test_ast_init_without_sorries()` - Verifies backward compatibility
- `test_ast_with_sorries_extracts_types()` - Verifies type extraction works

Run tests:
```bash
uv run pytest tests/test_parsers_ast.py -v
```

## Benefits

1. **Complete Lemmas**: Generated subgoal lemmas now include all necessary type information
2. **Standalone Code**: Extracted subgoals can be compiled independently without context
3. **Better Type Safety**: Explicit type declarations prevent ambiguity
4. **Backward Compatible**: Existing code continues to work without modification
5. **Accurate Context**: Preserves the exact type context from the proof state

## Example Output Comparison

### Input Lean Code
```lean
theorem u31 (O A C B D : ℂ) ... := by
  have hOA_parallel_CA_or : C = A ∨ ∃ r : ℝ, O - A = r • (C - A) := by
    sorry
```

### Old Output (Without Sorries)
```lean
lemma hOA_parallel_CA_or : C = A ∨ ∃ r, O - A = r • (C - A) := by sorry
```
❌ Missing type declarations - won't compile standalone

### New Output (With Sorries)
```lean
lemma hOA_parallel_CA_or (O : ℂ) (A : ℂ) (C : ℂ) : C = A ∨ ∃ r : ℝ, O - A = r • (C - A) := by sorry
```
✅ Complete type information - compiles standalone

## Notes

- The goal parsing handles Lean 4's Unicode symbols (ℂ, ℕ, ℝ, ¬, ∨, ∃, etc.)
- Multiple variables with the same type are correctly parsed (e.g., "O A C B D : ℂ")
- Complex type expressions are preserved (e.g., "((O - A) / (C - A)).im = 0")
- The implementation prioritizes information from sorries over AST-derived types when available
