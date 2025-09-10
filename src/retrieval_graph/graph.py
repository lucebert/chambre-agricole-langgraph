### Nodes

from langchain import hub
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph

from retrieval_graph.configuration import RetreiveConfiguration
from retrieval_graph.state import GraphState, InputState
from shared.retrieval import make_pinecone_retriever


def retrieve(state: GraphState) -> dict[str, list[str] | str]: 
    """Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    # Extract human messages and concatenate them
    question = " ".join(msg.content for msg in state.messages if isinstance(msg, HumanMessage))

    # Retrieval
    with make_pinecone_retriever(OpenAIEmbeddings(model="text-embedding-3-small")) as retriever:
        documents = retriever.invoke(question)
        return {"documents": documents, "message": state.messages}


async def generate(state: GraphState):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    messages = state.messages
    documents = state.documents

    # RAG generation
    # Prompt
    prompt = hub.pull("chambre-agri/rag-main")

    # LLM
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    

    # Chain
    rag_chain = prompt + messages | llm
    response = await rag_chain.ainvoke({"context" : documents})
    return {"messages": [response], "documents": documents}


workflow = StateGraph(GraphState, input_schema=InputState)

# Define the nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

# Build graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile
graph = workflow.compile()
graph.name = "RetrievalGraph"
