"""Define the configurable parameters for the index graph."""

from __future__ import annotations

from dataclasses import dataclass

from shared.configuration import BaseConfiguration

# This file contains sample documents to index, based on the following LangChain and LangGraph documentation pages:
# - https://python.langchain.com/v0.3/docs/concepts/
# - https://langchain-ai.github.io/langgraph/concepts/low_level/


@dataclass(kw_only=True)
class RetreiveConfiguration(BaseConfiguration):
    """Configuration class for indexing and retrieval operations.

    This class defines the parameters needed for configuring the indexing and
    retrieval processes, including embedding model selection, retriever provider choice, and search parameters.
    """
    retreive_model: str = "gpt-4o"