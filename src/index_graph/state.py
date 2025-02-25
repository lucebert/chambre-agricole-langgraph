"""State management for the index graph."""

from dataclasses import dataclass, field

# The index state defines the simple IO for the single-node index graph
@dataclass(kw_only=True)
class InputState:
    """Represents the state for document indexing and retrieval.

    This class defines the structure of the index state, which includes
    the documents to be indexed and the retriever used for searching
    these documents.
    """
    title: str
    publication_year: str
    publisher: str
    url: str
    project_code: str

@dataclass(kw_only=True)
class IndexState(InputState):
    """Represents the state for document indexing and retrieval.

    This class defines the structure of the index state, which includes
    the documents to be indexed and the retriever used for searching
    these documents.
    """
    pdf_text: str = field(default_factory=str)
    metadata: dict = field(default_factory=dict)