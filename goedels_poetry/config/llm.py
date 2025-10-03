from typing import Any

from langchain_ollama import ChatOllama
from ollama import ResponseError, chat, pull
from rich.console import Console
from tqdm import tqdm

from goedels_poetry.config.config import parsed_config

# Create Console for outputs
console = Console()


def _download_llm(llm: str) -> None:
    """
    Method which ensures the specified LLM is downloaded. This code if based off of that provided
    # by Ollama https://github.com/ollama/ollama-python/blob/main/examples/pull.py

    Parameters
    ----------
    llm: str
        The LLM to ensure download of
    """
    # Inform user of progress
    console.print(f"Starting download of {llm}")

    # Track progress for each layer
    current_digest: str = ""
    bars: dict[str, Any] = {}
    for progress in pull(llm, stream=True):
        digest = progress.get("digest", "")
        if digest != current_digest and current_digest in bars:
            bars[current_digest].close()

        if not digest:
            console.print(progress.get("status"))
            continue

        if digest not in bars and (total := progress.get("total")):
            bars[digest] = tqdm(total=total, desc=f"pulling {digest[7:19]}", unit="B", unit_scale=True)

        if completed := progress.get("completed"):
            bars[digest].update(completed - bars[digest].n)

        current_digest = digest


def _download_llms(llms: list[str]) -> None:
    """
    Method which ensures the specified LLMs are downloaded.

    Parameters
    ----------
    llms: list[str]
        The LLMs to download
    """
    # Download the LLMs one at a time
    for llm in llms:
        try:
            # Check to see if it's already downloaded
            chat(llm)
        except ResponseError as e:
            # If it isn't downloaded, download it
            if e.status_code == 404:
                console.print(f"Starting download of {llm}")
                _download_llm(llm)


# LLMS to download
_LLMS = [
    parsed_config.get(section="FORMALIZER_AGENT_LLM", option="model", fallback="kdavis/goedel-formalizer-v2:32b"),
    parsed_config.get(section="PROVER_AGENT_LLM", option="model", fallback="kdavis/Goedel-Prover-V2:32b"),
    parsed_config.get(section="SEMANTICS_AGENT_LLM", option="model", fallback="qwen3:30b"),
    parsed_config.get(section="DECOMPOSER_AGENT_LLM", option="model", fallback="qwen3:30b"),
]


# Download LLMS
_download_llms(_LLMS)


# Create LLMS
FORMALIZER_AGENT_LLM = ChatOllama(
    model=parsed_config.get(section="FORMALIZER_AGENT_LLM", option="model", fallback="kdavis/goedel-formalizer-v2:32b"),
    validate_model_on_init=True,
    num_predict=50000,
    num_ctx=parsed_config.getint(section="FORMALIZER_AGENT_LLM", option="num_ctx", fallback=40960),
)
PROVER_AGENT_LLM = ChatOllama(
    model=parsed_config.get(section="PROVER_AGENT_LLM", option="model", fallback="kdavis/Goedel-Prover-V2:32b"),
    validate_model_on_init=True,
    num_predict=50000,
    num_ctx=parsed_config.getint(section="PROVER_AGENT_LLM", option="num_ctx", fallback=40960),
)
SEMANTICS_AGENT_LLM = ChatOllama(
    model=parsed_config.get(section="SEMANTICS_AGENT_LLM", option="model", fallback="qwen3:30b"),
    validate_model_on_init=True,
    num_predict=50000,
    num_ctx=parsed_config.getint(section="SEMANTICS_AGENT_LLM", option="num_ctx", fallback=262144),
)
DECOMPOSER_AGENT_LLM = ChatOllama(
    model=parsed_config.get(section="DECOMPOSER_AGENT_LLM", option="model", fallback="qwen3:30b"),
    validate_model_on_init=True,
    num_predict=50000,
    num_ctx=parsed_config.getint(section="DECOMPOSER_AGENT_LLM", option="num_ctx", fallback=262144),
)

# Create LLM configurations
PROVER_AGENT_MAX_RETRIES = parsed_config.getint(section="PROVER_AGENT_LLM", option="max_retries", fallback=10)
PROVER_AGENT_MAX_DEPTH = parsed_config.getint(section="PROVER_AGENT_LLM", option="max_depth", fallback=20)
DECOMPOSER_AGENT_MAX_RETRIES = parsed_config.getint(section="DECOMPOSER_AGENT_LLM", option="max_retries", fallback=5)
FORMALIZER_AGENT_MAX_RETRIES = parsed_config.getint(section="FORMALIZER_AGENT_LLM", option="max_retries", fallback=10)
