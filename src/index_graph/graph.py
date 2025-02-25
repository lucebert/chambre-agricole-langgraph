"""This "graph" simply exposes an endpoint for a user to upload docs to be indexed."""
from typing import Optional

from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

from index_graph.configuration import IndexConfiguration
from index_graph.pdf_parser import PDFParser
from index_graph.state import IndexState, InputState
from shared import retrieval


async def retreive_pdf(
    state: InputState, *, config: Optional[RunnableConfig] = None
) -> dict[str, str]:
    """Retrieve the PDF from the URL."""
    pdf_parser = PDFParser()
    
    text = await pdf_parser.process_pdf(state.url)
    
    metadata = {
            "title": state.title,
            "publication_year": state.publication_year,
            "publisher": state.publisher,
            "url": state.url,
            "project_code": state.project_code,
    }

    if text:
        return {"metadata": metadata, "pdf_text": text}
    else:
        return {}


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
    if state.pdf_text:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.create_documents([state.pdf_text], metadatas=[state.metadata])
        with retrieval.make_retriever(config) as retriever:
                await retriever.aadd_documents(docs)
                print(
                    f"Indexed docs {docs}"
                )

    # URL OK, intégrer index
    return {}

# Define the graph
builder = StateGraph(IndexState, input=InputState, config_schema=IndexConfiguration)
builder.add_node(retreive_pdf)
builder.add_node(index_docs)

builder.add_edge(START, "retreive_pdf")
builder.add_edge("retreive_pdf", "index_docs")
builder.add_edge("index_docs", END)
# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "IndexGraph"
