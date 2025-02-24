"""This "graph" simply exposes an endpoint for a user to upload docs to be indexed."""

from typing import List, Optional

import requests
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

from index_graph.configuration import IndexConfiguration
from index_graph.loaders import CodeProjectLoader
from index_graph.pdf_parser import PDFParser
from index_graph.state import IndexState
from shared import retrieval


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
    # if not config:
    #     raise ValueError("Configuration required to run index_docs.")

    # project_list = state.project_list

    # Process each URL
    # docs = []
    print(state.fullText)

    payload = {
            "fullText": "couverts végétaux",
            "motsCles": [],
            "enrichPath": [
                "region_complexe/Normandie"
            ],
            "organismes": [],
            "typesDonnees": ["DOCUMENT"],
            "annees": [],
            "page": 0,
            "searchSize": 2000,
            "tri": "DATE",
        }
    
    response = requests.post("https://rd-agri.fr/rest/search/getResults", json=payload, verify=False)
    data = response.json()

    pdf_parser = PDFParser()

    for document in data.get("results", []):
        urlDocument = document.get("urlDocument")
        if (urlDocument is None or not urlDocument.split('?')[0].lower().endswith('.pdf')): #or urlDocument.startswith('/rest/content/getFile/')):
            continue
        if (urlDocument.startswith('/rest/content/getFile/')):
            urlDocument = 'https://rd-agri.fr' + urlDocument
        metadata = {
                "title": document.get("titre"),
                "publication_year": document.get("anneePublication"),
                "publisher": document.get("publicateur"),
                "url": urlDocument,
                "project_code": document.get("codeProjet"),
        }
        text = pdf_parser.process_pdf(urlDocument)
        if text:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.create_documents([text], metadatas=[metadata])
            with retrieval.make_retriever(config) as retriever:
                    await retriever.aadd_documents(docs)
                    print(
                        f"Indexed docs {docs}"
                    )

    # URL OK, intégrer index
    return {}


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
