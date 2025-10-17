"""Integration tests for agents that use KiminaClient.

These tests verify that agent factories properly interact with the Kimina Lean server.
They test the 6 agent factories that use KiminaClient but not BaseChatModel:
1. ProofCheckerAgentFactory
2. ProofParserAgentFactory
3. SketchCheckerAgentFactory
4. SketchParserAgentFactory
5. FormalTheoremSyntaxAgentFactory
6. InformalTheoremSyntaxAgentFactory
"""

import pytest

# Try to import the required modules - skip all tests if imports fail
# Note: These tests require a separate Kimina Lean Server installation
try:
    from goedels_poetry.agents.formal_theorem_syntax_agent import FormalTheoremSyntaxAgentFactory
    from goedels_poetry.agents.informal_theorem_syntax_agent import InformalTheoremSyntaxAgentFactory
    from goedels_poetry.agents.proof_checker_agent import ProofCheckerAgentFactory
    from goedels_poetry.agents.proof_parser_agent import ProofParserAgentFactory
    from goedels_poetry.agents.sketch_checker_agent import SketchCheckerAgentFactory
    from goedels_poetry.agents.sketch_parser_agent import SketchParserAgentFactory
    from goedels_poetry.agents.state import (
        DecomposedFormalTheoremState,
        DecomposedFormalTheoremStates,
        FormalTheoremProofState,
        FormalTheoremProofStates,
        InformalTheoremState,
    )

    IMPORTS_AVAILABLE = True
except (ImportError, TypeError) as e:
    # Skip tests if imports fail (e.g., Python < 3.10 with kimina-lean-server)
    IMPORTS_AVAILABLE = False
    SKIP_REASON = f"Failed to import required modules: {e}"

# Skip entire module if imports not available
if not IMPORTS_AVAILABLE:
    pytestmark = pytest.mark.skip(reason=SKIP_REASON)
else:
    # Mark all tests in this module as requiring Lean
    pytestmark = pytest.mark.usefixtures("skip_if_no_lean")


# Sample Lean code for testing
# Note: imports are handled by the agents, so we don't include them here
VALID_LEAN_CODE = """
theorem test_theorem : True := by
  trivial
"""

INVALID_LEAN_CODE = """
theorem broken : False := by
  trivial
"""

SIMPLE_THEOREM = """
theorem simple : 1 + 1 = 2 := by
  rfl
"""


class TestProofCheckerAgent:
    """Tests for ProofCheckerAgentFactory."""

    def test_create_agent(self, kimina_server_url: str) -> None:
        """Test that ProofCheckerAgent can be created."""
        agent = ProofCheckerAgentFactory.create_agent(server_url=kimina_server_url, server_max_retries=3)
        assert agent is not None

    def test_check_valid_proof(self, kimina_server_url: str) -> None:
        """Test checking a valid proof."""
        agent = ProofCheckerAgentFactory.create_agent(server_url=kimina_server_url, server_max_retries=3)

        # Create input state
        state: FormalTheoremProofState = {
            "formal_theorem": SIMPLE_THEOREM,
            "formal_proof": VALID_LEAN_CODE,
            "proved": False,
            "errors": "",
            "ast": None,
        }
        input_states: FormalTheoremProofStates = {"inputs": [state], "outputs": []}

        # Run agent
        result = agent.invoke(input_states)

        # Verify result
        assert "outputs" in result
        assert len(result["outputs"]) == 1
        output_state = result["outputs"][0]
        assert output_state["proved"] is True
        assert output_state["errors"] == ""

    def test_check_invalid_proof(self, kimina_server_url: str) -> None:
        """Test checking an invalid proof."""
        agent = ProofCheckerAgentFactory.create_agent(server_url=kimina_server_url, server_max_retries=3)

        # Create input state with invalid proof
        state: FormalTheoremProofState = {
            "formal_theorem": SIMPLE_THEOREM,
            "formal_proof": INVALID_LEAN_CODE,
            "proved": False,
            "errors": "",
            "ast": None,
        }
        input_states: FormalTheoremProofStates = {"inputs": [state], "outputs": []}

        # Run agent
        result = agent.invoke(input_states)

        # Verify result
        assert "outputs" in result
        assert len(result["outputs"]) == 1
        output_state = result["outputs"][0]
        assert output_state["proved"] is False
        assert len(output_state["errors"]) > 0


class TestProofParserAgent:
    """Tests for ProofParserAgentFactory."""

    def test_create_agent(self, kimina_server_url: str) -> None:
        """Test that ProofParserAgent can be created."""
        agent = ProofParserAgentFactory.create_agent(server_url=kimina_server_url, server_max_retries=3)
        assert agent is not None

    def test_parse_proof(self, kimina_server_url: str) -> None:
        """Test parsing a valid proof."""
        agent = ProofParserAgentFactory.create_agent(server_url=kimina_server_url, server_max_retries=3)

        # Create input state
        state: FormalTheoremProofState = {
            "formal_theorem": SIMPLE_THEOREM,
            "formal_proof": VALID_LEAN_CODE,
            "proved": False,
            "errors": "",
            "ast": None,
        }
        input_states: FormalTheoremProofStates = {"inputs": [state], "outputs": []}

        # Run agent
        result = agent.invoke(input_states)

        # Verify result
        assert "outputs" in result
        assert len(result["outputs"]) == 1
        output_state = result["outputs"][0]
        assert output_state["ast"] is not None


class TestSketchCheckerAgent:
    """Tests for SketchCheckerAgentFactory."""

    def test_create_agent(self, kimina_server_url: str) -> None:
        """Test that SketchCheckerAgent can be created."""
        agent = SketchCheckerAgentFactory.create_agent(server_url=kimina_server_url, server_max_retries=3)
        assert agent is not None

    def test_check_valid_sketch(self, kimina_server_url: str) -> None:
        """Test checking a valid proof sketch."""
        agent = SketchCheckerAgentFactory.create_agent(server_url=kimina_server_url, server_max_retries=3)

        # Create input state
        state: DecomposedFormalTheoremState = {
            "formal_theorem": SIMPLE_THEOREM,
            "proof_sketch": VALID_LEAN_CODE,
            "syntactic": False,
            "errors": "",
            "ast": None,
        }
        input_states: DecomposedFormalTheoremStates = {"inputs": [state], "outputs": []}

        # Run agent
        result = agent.invoke(input_states)

        # Verify result
        assert "outputs" in result
        assert len(result["outputs"]) == 1
        output_state = result["outputs"][0]
        assert output_state["syntactic"] is True
        assert output_state["errors"] == ""

    def test_check_invalid_sketch(self, kimina_server_url: str) -> None:
        """Test checking an invalid proof sketch."""
        agent = SketchCheckerAgentFactory.create_agent(server_url=kimina_server_url, server_max_retries=3)

        # Create input state with invalid sketch
        state: DecomposedFormalTheoremState = {
            "formal_theorem": SIMPLE_THEOREM,
            "proof_sketch": INVALID_LEAN_CODE,
            "syntactic": False,
            "errors": "",
            "ast": None,
        }
        input_states: DecomposedFormalTheoremStates = {"inputs": [state], "outputs": []}

        # Run agent
        result = agent.invoke(input_states)

        # Verify result
        assert "outputs" in result
        assert len(result["outputs"]) == 1
        output_state = result["outputs"][0]
        assert output_state["syntactic"] is False
        assert len(output_state["errors"]) > 0


class TestSketchParserAgent:
    """Tests for SketchParserAgentFactory."""

    def test_create_agent(self, kimina_server_url: str) -> None:
        """Test that SketchParserAgent can be created."""
        agent = SketchParserAgentFactory.create_agent(server_url=kimina_server_url, server_max_retries=3)
        assert agent is not None

    def test_parse_sketch(self, kimina_server_url: str) -> None:
        """Test parsing a valid proof sketch."""
        agent = SketchParserAgentFactory.create_agent(server_url=kimina_server_url, server_max_retries=3)

        # Create input state
        state: DecomposedFormalTheoremState = {
            "formal_theorem": SIMPLE_THEOREM,
            "proof_sketch": VALID_LEAN_CODE,
            "syntactic": False,
            "errors": "",
            "ast": None,
        }
        input_states: DecomposedFormalTheoremStates = {"inputs": [state], "outputs": []}

        # Run agent
        result = agent.invoke(input_states)

        # Verify result
        assert "outputs" in result
        assert len(result["outputs"]) == 1
        output_state = result["outputs"][0]
        assert output_state["ast"] is not None


class TestFormalTheoremSyntaxAgent:
    """Tests for FormalTheoremSyntaxAgentFactory."""

    def test_create_agent(self, kimina_server_url: str) -> None:
        """Test that FormalTheoremSyntaxAgent can be created."""
        agent = FormalTheoremSyntaxAgentFactory.create_agent(server_url=kimina_server_url, server_max_retries=3)
        assert agent is not None

    def test_check_valid_theorem(self, kimina_server_url: str) -> None:
        """Test checking a valid formal theorem."""
        agent = FormalTheoremSyntaxAgentFactory.create_agent(server_url=kimina_server_url, server_max_retries=3)

        # Create input state
        state: FormalTheoremProofState = {
            "formal_theorem": VALID_LEAN_CODE,
            "formal_proof": "",
            "proved": False,
            "errors": "",
            "ast": None,
        }
        input_states: FormalTheoremProofStates = {"inputs": [state], "outputs": []}

        # Run agent
        result = agent.invoke(input_states)

        # Verify result
        assert "outputs" in result
        assert len(result["outputs"]) == 1
        output_state = result["outputs"][0]
        assert output_state["syntactic"] is True

    def test_check_invalid_theorem(self, kimina_server_url: str) -> None:
        """Test checking an invalid formal theorem."""
        agent = FormalTheoremSyntaxAgentFactory.create_agent(server_url=kimina_server_url, server_max_retries=3)

        # Create input state with invalid theorem
        state: FormalTheoremProofState = {
            "formal_theorem": INVALID_LEAN_CODE,
            "formal_proof": "",
            "proved": False,
            "errors": "",
            "ast": None,
        }
        input_states: FormalTheoremProofStates = {"inputs": [state], "outputs": []}

        # Run agent
        result = agent.invoke(input_states)

        # Verify result
        assert "outputs" in result
        assert len(result["outputs"]) == 1
        output_state = result["outputs"][0]
        assert output_state["syntactic"] is False


class TestInformalTheoremSyntaxAgent:
    """Tests for InformalTheoremSyntaxAgentFactory."""

    def test_create_agent(self, kimina_server_url: str) -> None:
        """Test that InformalTheoremSyntaxAgent can be created."""
        agent = InformalTheoremSyntaxAgentFactory.create_agent(server_url=kimina_server_url, server_max_retries=3)
        assert agent is not None

    def test_check_valid_theorem(self, kimina_server_url: str) -> None:
        """Test checking a valid formal theorem from informal state."""
        agent = InformalTheoremSyntaxAgentFactory.create_agent(server_url=kimina_server_url, server_max_retries=3)

        # Create input state
        state: InformalTheoremState = {
            "informal_theorem": "Prove that true is true",
            "formal_theorem": VALID_LEAN_CODE,
            "syntactic": False,
            "semantic": False,
        }

        # Run agent
        result = agent.invoke(state)

        # Verify result
        assert result["syntactic"] is True

    def test_check_invalid_theorem(self, kimina_server_url: str) -> None:
        """Test checking an invalid formal theorem from informal state."""
        agent = InformalTheoremSyntaxAgentFactory.create_agent(server_url=kimina_server_url, server_max_retries=3)

        # Create input state with invalid theorem
        state: InformalTheoremState = {
            "informal_theorem": "Prove that false is true",
            "formal_theorem": INVALID_LEAN_CODE,
            "syntactic": False,
            "semantic": False,
        }

        # Run agent
        result = agent.invoke(state)

        # Verify result
        assert result["syntactic"] is False
