# AST.get_named_subgoal_code() Enhancement

## Overview

Enhanced the `AST.get_named_subgoal_code()` method to generate **standalone lemmas** that can be proven independently without reference to their enclosing theorem.

## What Changed

The method now includes:

1. **Enclosing theorem's parameters and hypotheses** - All type parameters and assumptions from the parent theorem are now added as parameters to the extracted lemma
2. **Earlier have statements** - Have statements that appear textually before the target subgoal are included as additional hypotheses
3. **Complete type information** - Improved type extraction to correctly parse and include the full type signature

## Example

Given a theorem:

```lean
theorem u31 (O A C B D : ℂ) (hd₁ : ¬B = D) (hd₂ : ¬C = D) : Prop := by
  have hCD_ne : C - D ≠ 0 := by sorry
  have hDB_ne : D - B ≠ 0 := by sorry
  have hCB_ne : C ≠ B := by sorry
```

### Before Enhancement

Calling `get_named_subgoal_code("hDB_ne")` would return:
```lean
lemma hDB_ne : D - B ≠ 0 := by sorry
```

This lemma **cannot be proven** standalone because:
- Missing theorem parameters (O, A, C, B, D : ℂ)
- Missing theorem hypotheses (hd₁, hd₂)
- Missing earlier have statement (hCD_ne)

### After Enhancement

Now returns:
```lean
lemma hDB_ne (O A C B D : ℂ) (hd₁ : ¬B = D) (hd₂ : ¬C = D) (hCD_ne : C - D ≠ 0) : D - B ≠ 0 := by sorry
```

This lemma **can be proven** standalone with all necessary context!

## Implementation Details

### New Helper Functions

1. **`__find_enclosing_theorem()`** - Locates the theorem/lemma that contains a given subgoal
2. **`__extract_theorem_binders()`** - Extracts all parameters and hypotheses from a theorem
3. **`__find_earlier_haves()`** - Finds have statements that appear textually before the target
4. **`__extract_binder_name()`** - Extracts variable names from binder AST nodes

### Modified Functions

1. **`__extract_type_ast()`** - Improved to correctly extract type specifications from have statements by collecting all tokens between `:` and `:=`
2. **`__strip_leading_colon()`** - Updated to handle the new `__type_container` wrapper for type tokens
3. **`_get_named_subgoal_rewritten_ast()`** - Enhanced to:
   - Find and include enclosing theorem's binders
   - Collect and include earlier have statements
   - Deduplicate variables to avoid redundant parameters
   - Merge type information from all available sorry entries

## Testing

Added comprehensive tests in `tests/test_parsers_ast.py`:

- `test_ast_get_named_subgoal_code_includes_theorem_hypotheses()` - Verifies theorem parameters and hypotheses are included
- `test_ast_get_named_subgoal_code_includes_earlier_haves()` - Verifies earlier have statements are included with correct types

All existing tests continue to pass, confirming backward compatibility.

## Benefits

1. **Standalone Lemmas** - Extracted subgoals can now be proven independently
2. **Better Context** - Provers have access to all necessary hypotheses and type information
3. **Modular Proving** - Subgoals can be worked on in isolation without needing the full theorem
4. **Improved Debugging** - Clearer understanding of what each subgoal requires

## Files Modified

- `goedels_poetry/parsers/util.py` - Core implementation
- `tests/test_parsers_ast.py` - Added comprehensive tests

## Backward Compatibility

The enhancement is fully backward compatible. Existing code using `get_named_subgoal_code()` will automatically benefit from the improved output without any changes required.

## Further Enhancements

See [LET_AND_OBTAIN_ENHANCEMENT.md](LET_AND_OBTAIN_ENHANCEMENT.md) for the follow-up enhancement that adds support for `let` bindings and `obtain` statements, making lemmas even more independently provable.
