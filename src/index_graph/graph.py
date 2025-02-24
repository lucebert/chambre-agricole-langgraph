"""This "graph" simply exposes an endpoint for a user to upload docs to be indexed."""

import asyncio
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

from index_graph.configuration import IndexConfiguration
from index_graph.loaders import CodeProjectLoader
from index_graph.state import IndexState


async def index_docs(
    state: IndexState, *, config: Optional[RunnableConfig] = None
) -> dict[str, str]:
    """Asynchronously index documents in the given state using the configured retriever.

    This function takes the documents from the state, ensures they have a user ID,
    adds them to the retriever's index, and then signals for the documents to be
    deleted from the state.

    If docs are not provided in the state, they will be loaded
    from the configuration.docs_file JSON file.

    Args:
        state (IndexState): The current state containing documents and retriever.
        config (Optional[RunnableConfig]): Configuration for the indexing process.r
    """
    if not config:
        raise ValueError("Configuration required to run index_docs.")

    project_list = state.project_list

    # Process each URL
    docs = []

    # Convert synchronous function to async
    async def async_get_code_chunks(code):
        return await asyncio.get_event_loop().run_in_executor(
            None, get_chuncks_from_code_project, code
        )

    # Process all URLs in parallel
    chunk_tasks = [async_get_code_chunks(code) for code in project_list]
    chunks_list = await asyncio.gather(*chunk_tasks)

    # Flatten the list of lists into single docs list
    docs = [doc for chunks in chunks_list for doc in chunks]

    # with retrieval.make_retriever(config) as retriever:
    #     batch_size = min(
    #         500, max(1, len(docs))
    #     )  # Target 500 docs per batch, but handle smaller doc counts
    #     for i in range(0, len(docs), batch_size):
    #         batch = docs[i : i + batch_size]
    #         await retriever.aadd_documents(batch)
    #         print(
    #             f"Indexed batch {i//batch_size + 1} of {(len(docs) + batch_size - 1)//batch_size}"
    #         )

    return {"docs": docs}


def get_chuncks_from_code_project(code: str) -> List[Document]:
    """Index a code project."""
    loader = CodeProjectLoader(
        project_code=code,
    )
    docs = loader.load()

    print(f"Loaded {len(docs)} documents from {code}.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    docs = text_splitter.split_documents(docs)

    print(f"Split {len(docs)} documents into chunks.")
    return docs


# Define the graph
builder = StateGraph(IndexState, config_schema=IndexConfiguration)
builder.add_node(index_docs)
builder.add_edge(START, "index_docs")
builder.add_edge("index_docs", END)
# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "IndexGraph"
