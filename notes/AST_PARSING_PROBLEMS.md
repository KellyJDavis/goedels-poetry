# AST Parsing Problems and Criticality Analysis

This document identifies problems in the AST parsing implementation that were not addressed in `AST_PARSING_DEEP_DIVE.md`, along with an assessment of their criticality.

---

## Problem 1: Unsafe Array Indexing Without Bounds Checking

### Status: **✅ FIXED**

### Description

The parser frequently accessed array elements without checking if the array has enough elements. This has been corrected by adding comprehensive bounds checking throughout the codebase.

### Original Issue

```python
# Before (unsafe):
decl_id = node["args"][1]  # declId
name = decl_id["args"][0]["val"]

have_decl = node["args"][1]  # Term.haveDecl
have_id_decl = have_decl["args"][0]
have_id = have_id_decl["args"][0]["args"][0]["val"]
```

### Fix Applied

All unsafe array indexing has been replaced with bounds checking:

```python
# After (safe):
node_args = node.get("args", [])
if len(node_args) > 1 and isinstance(node_args[1], dict):
    decl_id = node_args[1]
    decl_id_args = decl_id.get("args", [])
    if len(decl_id_args) > 0 and isinstance(decl_id_args[0], dict):
        name_node = decl_id_args[0]
        if "val" in name_node:
            name = name_node["val"]
```

### Functions Fixed

1. **`_context_after_decl()`** - Added bounds checking for declId extraction
2. **`_context_after_have()`** - Added bounds checking for 4-level deep nesting
3. **`_get_named_subgoal_ast()`** - Fixed theorem/lemma and have statement extraction
4. **`__contains_target_name()`** - Fixed have statement name extraction
5. **`__find_enclosing_theorem()`** - Fixed theorem/lemma and have statement extraction in `contains_target()`
6. **`__find_earlier_bindings()`** - Fixed have statement extraction in `traverse_for_bindings()`

### Impact

- **✅ No more IndexError** on malformed ASTs
- **✅ Better error handling** - failures are handled gracefully
- **✅ More robust** - handles AST structure variations
- **✅ Type safety** - checks isinstance() before accessing nested structures

### Criticality: **MEDIUM-HIGH** (Now Resolved)

---

## Problem 2: Overly Broad Exception Handling

### Status: **✅ FIXED**

### Description

The parser was using `except Exception:` which caught all exceptions, including programming errors. This has been corrected by using specific exception types throughout the codebase.

### Original Issue

```python
# Before (overly broad):
try:
    decl_id = node["args"][1]
    name = decl_id["args"][0]["val"]
    if name == target_name:
        return node
except Exception:
    logging.exception("Exception occurred")
```

### Fix Applied

All `except Exception:` handlers have been replaced with specific exception types:

```python
# After (specific):
try:
    node_args = node.get("args", [])
    if len(node_args) > 1 and isinstance(node_args[1], dict):
        decl_id = node_args[1]
        # ... safe access ...
except (KeyError, IndexError, TypeError, AttributeError) as e:
    logging.debug(f"Failed to extract name from {kind}: {e}")
```

### Exception Types Used

The parser now catches only the specific exceptions that can occur during AST parsing:
- **KeyError**: When accessing dictionary keys that don't exist
- **IndexError**: When accessing list indices that don't exist (defensive, though bounds checking should prevent most)
- **TypeError**: When operations are performed on wrong types
- **AttributeError**: When accessing attributes that don't exist

### Functions Fixed

1. **`_get_named_subgoal_ast()`** - Fixed 2 exception handlers
2. **`__extract_type_ast()`** - Fixed 4 exception handlers (obtain, choose, generalize, match)
3. **`__contains_target_name()`** - Fixed 1 exception handler
4. **`__find_enclosing_theorem()`** - Fixed 2 exception handlers in `contains_target()`
5. **`__find_earlier_bindings()`** - Fixed 10 exception handlers in `traverse_for_bindings()`
6. **`__extract_match_names()`** - Fixed 1 exception handler in `find_match_alts()`

### Impact

- **✅ Better debugging** - Specific exception types make it easier to identify problems
- **✅ No masking of programming errors** - Real bugs (like typos) will now surface
- **✅ More informative logging** - Exception details are included in log messages
- **✅ Better error handling** - Only expected exceptions are caught

### Criticality: **MEDIUM** (Now Resolved)

---

## Problem 3: No Input Validation

### Status: **✅ FIXED**

### Description

The parser was assuming the AST structure is valid without validating input. This has been corrected by adding comprehensive input validation at key entry points.

### Original Issue

```python
# Before (no validation):
def _get_named_subgoal_rewritten_ast(
    ast: Node, target_name: str, sorries: Optional[list[dict[str, Any]]] = None
) -> dict:
    name_map = __collect_named_decls(ast)
    if target_name not in name_map:
        raise KeyError(f"target '{target_name}' not found in AST")
    # No validation that ast is a valid AST structure
```

### Fix Applied

Added comprehensive input validation:

1. **Created `_validate_ast_structure()` function** - Validates AST structure:
   - Checks that AST is a dict or list (not None or other types)
   - Validates top-level structure (header/commands for AstExport format)
   - Validates list elements are valid nodes
   - Provides clear error messages

2. **Added validation in `AST.__init__()`** - Validates AST structure when creating AST object:
   ```python
   def __init__(self, ast: dict[str, Any], sorries: Optional[list[dict[str, Any]]] = None):
       from goedels_poetry.parsers.util import _validate_ast_structure
       # Validate AST structure
       if not _validate_ast_structure(ast, raise_on_error=True):
           raise ValueError("Invalid AST structure provided")
   ```

3. **Added validation in `_get_named_subgoal_rewritten_ast()`** - Validates:
   - AST structure
   - target_name is a non-empty string
   - sorries is a list of dicts (if provided)

4. **Added validation in `_get_named_subgoal_ast()`** - Validates target_name

### Validation Function

```python
def _validate_ast_structure(ast: Node, raise_on_error: bool = False) -> bool:
    """Validate that the AST has a basic valid structure.

    The AST from kimina-lean-server can be:
    - A top-level dict with "header" and "commands" fields (from AstExport.lean)
    - A dict representing a single node with "kind" field
    - A list of nodes
    - Any nested combination of the above
    """
    if ast is None:
        if raise_on_error:
            raise ValueError("AST cannot be None")
        return False

    # AST must be a dict or list
    if not isinstance(ast, (dict, list)):
        if raise_on_error:
            raise ValueError(f"AST must be a dict or list, got {type(ast).__name__}")
        return False

    # Validate structure based on type
    # ... (detailed validation logic)
```

### Functions Enhanced

1. **`AST.__init__()`** - Validates AST structure on initialization
2. **`_get_named_subgoal_rewritten_ast()`** - Validates AST, target_name, and sorries
3. **`_get_named_subgoal_ast()`** - Validates target_name
4. **`_validate_ast_structure()`** - New validation function

### Impact

- **✅ Early error detection** - Invalid input is caught immediately
- **✅ Better error messages** - Clear, specific error messages for invalid input
- **✅ Easier debugging** - Errors occur at expected places with helpful messages
- **✅ Type safety** - Validates types before processing

### Validation Checks

1. **AST structure**:
   - Not None
   - Is dict or list
   - Top-level has expected structure (header/commands or kind field)
   - List elements are valid nodes

2. **target_name**:
   - Is a string
   - Is non-empty

3. **sorries** (if provided):
   - Is a list
   - All elements are dicts

### Criticality: **MEDIUM** (Now Resolved)

---

## Problem 4: Goal Context Parsing Edge Cases

### Status: **✅ FIXED**

### Description

The `__parse_goal_context()` function had several edge cases that weren't handled correctly. These have been fixed with improved parsing logic and validation.

### Original Issues

1. **Colon in type**: If type contains a colon, parsing could fail
2. **Empty names**: If `names_part` is empty after stripping, `names.split()` returns `['']`
3. **Whitespace-only names**: Names with only whitespace were treated as valid
4. **No validation**: Didn't check if names are valid after processing

### Original Code

```python
# Before (edge cases not handled):
parts = line.rsplit(":", 1)
if len(parts) != 2:
    continue

names_part = parts[0].strip()
type_part = parts[1].strip()

# Handle multiple variables with same type
names = names_part.split()
for name in names:
    var_types[name] = type_part  # Could add empty/whitespace names
```

### Fix Applied

Improved parsing with comprehensive edge case handling:

```python
# After (robust handling):
# Validate input
if not isinstance(goal, str):
    return var_types

# Skip empty lines
if not line:
    continue

# Split at the last colon (handles colons in types)
parts = line.rsplit(":", 1)
if len(parts) != 2:
    continue

names_part = parts[0].strip()
type_part = parts[1].strip()

# Skip if no names or no type
if not names_part or not type_part:
    continue

# Filter out empty strings and whitespace-only strings
names = [n.strip() for n in names_part.split() if n.strip()]

# Validate names are non-empty after filtering
if not names:
    continue

# Add each valid name with its type
for name in names:
    var_types[name] = type_part
```

### Improvements

1. **Input validation**: Checks that `goal` is a string
2. **Empty line handling**: Skips empty lines explicitly
3. **Empty name/type handling**: Skips lines where names_part or type_part is empty
4. **Whitespace filtering**: Filters out empty and whitespace-only names using list comprehension
5. **Validation check**: Ensures names list is non-empty before processing
6. **Better comments**: Added documentation explaining the parsing strategy

### Edge Cases Handled

1. **Colon in type**: ✅ Using `rsplit(":", 1)` correctly handles this (splits from right)
2. **Empty names**: ✅ Filtered out with `if n.strip()` in list comprehension
3. **Whitespace-only names**: ✅ Filtered out with `if n.strip()` check
4. **Empty lines**: ✅ Explicitly skipped
5. **Missing names or types**: ✅ Validated before processing

### Notes on Colon Handling

The function uses `rsplit(":", 1)` which splits from the right at the last colon. This is correct for Lean goal context because:
- The format is always `names : type`
- The colon separating names from type is always the last colon in the line
- Even if the type contains colons (rare), `rsplit(":", 1)` correctly identifies the separator

### Criticality: **LOW-MEDIUM** (Now Resolved)

---

## Problem 5: Potential Infinite Loop in Hypothesis Name Generation

### Description

The `__generate_equality_hypothesis_name()` function has a safety limit, but it's very high:

```python
def __generate_equality_hypothesis_name(var_name: str, existing_names: set[str]) -> str:
    base_name = f"h{var_name}"
    if base_name not in existing_names:
        return base_name

    counter = 2
    while True:
        candidate = f"h{counter}{var_name}"
        if candidate not in existing_names:
            return candidate
        counter += 1
        # Safety limit to avoid infinite loops
        if counter > 1000:
            logging.warning(f"Could not generate unique hypothesis name for '{var_name}' after 1000 attempts")
            return f"h{counter}{var_name}"
```

### Issues

1. **Very high limit**: 1000 iterations is excessive - if this happens, there's a real problem
2. **Still returns non-unique name**: After 1000 attempts, returns a name that may still conflict
3. **No better strategy**: Doesn't try alternative naming schemes

### Criticality: **LOW**

**Why Not Critical:**
- Very unlikely to occur in practice
- Has a safety limit (though high)
- Would indicate a deeper problem if it happens

### Recommendation

Lower the limit and improve strategy:
```python
MAX_ATTEMPTS = 100  # More reasonable limit

if counter > MAX_ATTEMPTS:
    # Try UUID-based name as last resort
    import uuid
    unique_suffix = str(uuid.uuid4())[:8]
    return f"h{var_name}_{unique_suffix}"
```

---

## Problem 6: Deep Nesting Assumptions

### Status: **✅ FIXED**

### Description

The parser was assuming very specific AST structure depths without validation. This has been fixed by creating helper functions with documented structure assumptions based on Lean's parser grammar.

### Original Issue

```python
# Before (unsafe, undocumented):
have_id = have_id_decl["args"][0]["args"][0]["val"]
decl_id = node["args"][1]
name = decl_id["args"][0]["val"]
```

### Fix Applied

1. **Created `_extract_nested_value()` helper function** - Safely extracts values from nested structures
2. **Created `_extract_decl_id_name()` helper function** - Extracts theorem/lemma names with documented structure
3. **Created `_extract_have_id_name()` helper function** - Extracts have statement names with documented structure
4. **Replaced all deep nesting patterns** with helper function calls
5. **Added comprehensive documentation** explaining the AST structure based on Lean's parser grammar

### Structure Documentation

All structure assumptions are now documented with their origin:

#### 1. Theorem/Lemma Name Extraction (`_extract_decl_id_name`)

**Structure:**
- `theorem/lemma` node: `{"kind": "Lean.Parser.Command.theorem", "args": [..., declId, ...]}`
- `declId`: `{"kind": "Lean.Parser.Command.declId", "args": [name_node, ...]}`
- `name_node`: `{"val": "theorem_name", "info": {...}}` (from Syntax.ident or Syntax.atom)

**Origin:** Based on Lean's parser grammar where:
- `declId` is the first argument after the theorem/lemma keyword (args[1])
- The name is the first argument of `declId` (args[0])
- This matches the Lean parser structure: `theorem declId : type := proof`

**Path:** `["args", 1, "args", 0, "val"]`

#### 2. Have Statement Name Extraction (`_extract_have_id_name`)

**Structure:**
- `tacticHave_`: `{"kind": "Lean.Parser.Tactic.tacticHave_", "args": [..., haveDecl, ...]}`
- `haveDecl`: `{"kind": "Lean.Parser.Term.haveDecl", "args": [haveIdDecl, ...]}`
- `haveIdDecl`: `{"kind": "Lean.Parser.Term.haveIdDecl", "args": [haveId, ...]}`
- `haveId`: `{"kind": "Lean.Parser.Term.haveId", "args": [name_node, ...]}`
- `name_node`: `{"val": "have_name", "info": {...}}` (from Syntax.ident)

**Origin:** Based on Lean's parser grammar where:
- `haveDecl` is the second argument of `tacticHave_` (args[1])
- `haveIdDecl` is the first argument of `haveDecl` (args[0])
- `haveId` is the first argument of `haveIdDecl` (args[0])
- name is the first argument of `haveId` (args[0])
- This matches the Lean parser structure: `have haveIdDecl : type := proof`

**Path:** `["args", 1, "args", 0, "args", 0, "args", 0, "val"]`

### Helper Functions

#### `_extract_nested_value(node: dict, path: list[Union[int, str]], default: Any = None) -> Any`

Safely extracts values from nested structures using a path. Handles:
- Integer indices for lists
- String keys for dicts
- Bounds checking
- Type validation

**Based on:** Lean's Syntax AST structure from AstExport.lean:
- Syntax nodes: `{"kind": kind, "args": args, "info": info}`
- Atoms/Idents: `{"val": val, "info": info}`

#### `_extract_decl_id_name(node: dict[str, Any]) -> Optional[str]`

Extracts theorem/lemma names with full structure documentation.

#### `_extract_have_id_name(node: dict[str, Any]) -> Optional[str]`

Extracts have statement names with full structure documentation.

### Functions Refactored

1. **`_context_after_decl()`** - Now uses `_extract_decl_id_name()`
2. **`_context_after_have()`** - Now uses `_extract_have_id_name()`
3. **`_get_named_subgoal_ast()`** - Now uses helper functions
4. **`__contains_target_name()`** - Now uses `_extract_have_id_name()`
5. **`__find_enclosing_theorem()`** - Now uses helper functions in `contains_target()`
6. **`__find_earlier_bindings()`** - Now uses `_extract_have_id_name()` in `traverse_for_bindings()`

### Impact

- **✅ Safe extraction** - All nested access is now bounds-checked
- **✅ Documented structure** - Every assumption is documented with its origin
- **✅ Maintainable** - Helper functions make code easier to understand and modify
- **✅ Validated** - Structure validation happens before access
- **✅ Based on actual grammar** - All assumptions are based on Lean's parser grammar, not guesses

### Criticality: **MEDIUM** (Now Resolved)

---

## Problem 7: Silent Failures in Name Extraction

### Status: **✅ FIXED**

### Description

Many name extraction functions were returning `None` or empty lists silently when extraction failed. This has been fixed by adding comprehensive debug logging to all name extraction functions.

### Original Issue

```python
# Before (silent failures):
def __extract_let_name(let_node: dict) -> Optional[str]:
    let_id = __find_first(let_node, lambda n: n.get("kind") in {...})
    if let_id:
        val_node = __find_first(let_id, lambda n: isinstance(n.get("val"), str) and n.get("val") != "")
        if val_node:
            val = val_node.get("val")
            return str(val) if val is not None else None
    return None  # Silent failure - no indication why
```

### Fix Applied

Added comprehensive debug logging to all name extraction functions:

1. **`__extract_let_name()`** - Logs when:
   - Input is not a dict
   - letId/letIdDecl/binderIdent not found
   - val node not found
   - val is None

2. **`__extract_set_name()`** - Logs when:
   - Input is not a dict
   - setId/setIdDecl/binderIdent not found
   - val node not found
   - val is None

3. **`__extract_binder_name()`** - Logs when:
   - Input is not a dict
   - binderIdent not found
   - val node not found
   - val is None

4. **`__extract_obtain_names()`** - Logs when:
   - Input is not a dict
   - No names extracted (may be unnamed binding)

5. **`__extract_choose_names()`** - Logs when:
   - Input is not a dict
   - No names extracted (may be unnamed binding)

6. **`__extract_generalize_names()`** - Logs when:
   - Input is not a dict
   - No names extracted (may be unnamed binding)

7. **`__extract_match_names()`** - Logs when:
   - Input is not a dict
   - No names extracted (may be unnamed bindings or no matchAlt branches)

### Example Fix

```python
# After (with debug logging):
def __extract_let_name(let_node: dict) -> Optional[str]:
    """
    Extract the variable name from a let binding node.

    Returns None if the name cannot be extracted, with debug logging for failures.
    """
    if not isinstance(let_node, dict):
        logging.debug("__extract_let_name: let_node is not a dict")
        return None

    let_id = __find_first(let_node, lambda n: n.get("kind") in {...})
    if not let_id:
        logging.debug("__extract_let_name: Could not find letId/letIdDecl/binderIdent in let_node")
        return None

    val_node = __find_first(let_id, lambda n: isinstance(n.get("val"), str) and n.get("val") != "")
    if not val_node:
        logging.debug("__extract_let_name: Could not find val node with non-empty string in letId")
        return None

    val = val_node.get("val")
    if val is None:
        logging.debug("__extract_let_name: val node exists but val is None")
        return None

    return str(val)
```

### Functions Enhanced

1. **`__extract_let_name()`** - Added 4 debug log points
2. **`__extract_set_name()`** - Added 4 debug log points
3. **`__extract_binder_name()`** - Added 4 debug log points
4. **`__extract_obtain_names()`** - Added 2 debug log points
5. **`__extract_choose_names()`** - Added 2 debug log points
6. **`__extract_generalize_names()`** - Added 2 debug log points
7. **`__extract_match_names()`** - Added 2 debug log points

### Impact

- **✅ Better debugging** - Failures are now logged with specific reasons
- **✅ Easier troubleshooting** - Can identify why extraction failed
- **✅ Better visibility** - Invalid AST structures are now logged
- **✅ Maintains backward compatibility** - Still returns None/empty list, just with logging

### Logging Strategy

- **Debug level** - Used for all logging (won't spam production logs)
- **Specific messages** - Each failure point has a unique, descriptive message
- **Context-aware** - Messages indicate what was expected vs. what was found
- **Non-intrusive** - Logging doesn't change return values or behavior

### Criticality: **LOW-MEDIUM** (Now Resolved)

---

## Problem 8: No Type Validation for AST Nodes

### Description

The parser doesn't validate that nodes have expected types:

```python
def __extract_type_ast(node: Any, binding_name: Optional[str] = None) -> Optional[dict]:
    if not isinstance(node, dict):
        return None
    k = node.get("kind", "")
    # No validation that k is a string or that node has expected structure
```

### Impact

- **Type errors** may occur at runtime
- **No early detection** of invalid input
- **Poor error messages** when types are wrong

### Criticality: **LOW**

**Why Not Critical:**
- Basic type checks exist (`isinstance(node, dict)`)
- Most issues would be caught by Python's type system
- Not a major source of bugs

### Recommendation

Add more specific validation if needed:
```python
if not isinstance(node, dict):
    return None
if "kind" not in node:
    logging.warning("AST node missing 'kind' field")
    return None
```

---

## Problem 9: Circular Reference Risk in Deep Copy

### Description

The parser uses `deepcopy()` extensively, which could fail on circular references:

```python
target = deepcopy(name_map[target_name])
```

### Impact

- **RecursionError** if AST has circular references
- **Performance issues** with very large ASTs
- **Memory issues** with deep copying

### Criticality: **LOW**

**Why Not Critical:**
- Lean ASTs shouldn't have circular references
- `deepcopy` handles most cases well
- Would only be an issue with malformed ASTs

### Recommendation

Add error handling:
```python
try:
    target = deepcopy(name_map[target_name])
except RecursionError:
    logging.error("Circular reference detected in AST")
    raise
```

---

## Problem 10: Missing Validation of Generated Code

### Description

The parser generates Lean code but doesn't validate it:

```python
def get_named_subgoal_code(self, subgoal_name: str) -> str:
    rewritten_subgoal_ast = _get_named_subgoal_rewritten_ast(self._ast, subgoal_name, self._sorries)
    return str(_ast_to_code(rewritten_subgoal_ast))
    # No validation that generated code is valid Lean
```

### Impact

- **Invalid code** may be generated
- **Errors only surface** when code is checked by Lean
- **Hard to debug** - don't know if problem is in generation or elsewhere

### Criticality: **LOW**

**Why Not Critical:**
- Generated code is checked by Lean server anyway
- Validation would add complexity
- Not a blocker for functionality

### Recommendation

Optional validation (could be expensive):
```python
# Optional: validate generated code
if validate_generated_code:
    # Send to Lean server for quick syntax check
    pass
```

---

## Summary of Criticality

### High Priority (Should Fix)
- None identified

### Medium-High Priority (Should Fix Soon)
1. **Unsafe Array Indexing** - Can cause crashes, masks errors

### Medium Priority (Should Fix When Possible)
2. **Overly Broad Exception Handling** - Makes debugging harder
3. **No Input Validation** - Better error messages needed
6. **Deep Nesting Assumptions** - Code is brittle

### Low-Medium Priority (Nice to Have)
4. **Goal Context Parsing Edge Cases** - May cause issues in edge cases
7. **Silent Failures** - Better logging would help

### Low Priority (Minor Issues)
5. **Infinite Loop Risk** - Very unlikely
8. **No Type Validation** - Basic checks exist
9. **Circular Reference Risk** - Shouldn't occur
10. **Missing Code Validation** - Code is validated by Lean anyway

---

## Recommendations

### Immediate Actions
1. Add bounds checking for array access
2. Use specific exception types instead of `except Exception:`
3. Add helper functions for safe nested access

### Short-term Improvements
4. Add input validation
5. Improve error messages and logging
6. Document expected AST structures

### Long-term Enhancements
7. Add comprehensive test coverage for edge cases
8. Consider AST structure validation layer
9. Add optional generated code validation

---

## Conclusion

The AST parser is **generally robust** but has several areas for improvement:

1. **Safety**: Add bounds checking and better error handling
2. **Maintainability**: Reduce deep nesting assumptions, add documentation
3. **Debuggability**: Better logging, specific exception types
4. **Robustness**: Input validation, edge case handling

Most issues are **not critical** for current functionality but would improve code quality and maintainability. The highest priority is fixing unsafe array indexing, which could cause crashes on malformed ASTs.
