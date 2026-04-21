from typing import Annotated, TypedDict, NotRequired
import time

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama import ChatOllama, OllamaEmbeddings

# ==========================================
# 1. LOCAL RAG PIPELINE (100% Ollama)
# ==========================================
def setup_retriever():
    print("Initializing Local Knowledge Base...")
    kb_data = [
        "AutoStream Basic Plan Pricing & Features: $29/month. Includes 10 videos/month and 720p resolution.",
        "AutoStream Pro Plan Pricing & Features: $79/month. Includes Unlimited videos, 4K resolution, and AI captions.",
        "AutoStream Company Policy: No refunds after 7 days.",
        "AutoStream Company Policy: 24/7 support is available only on the Pro plan.",
    ]
    documents = [Document(page_content=text) for text in kb_data]
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 1})

retriever = setup_retriever()

# ==========================================
# 2. TOOLS & STATE DEFINITION
# ==========================================
@tool
def mock_lead_capture(name: str, email: str) -> str:
    """Use this tool ONLY when you have collected the user's REAL Name AND Email."""
    
    # Validation Guardrail: Prevent Llama from hallucinating fake emails
    if not name or not email or "<" in name or "@" not in email:
        return "ERROR: You gave invalid details. Ask the user for their actual name and email again."
        
    print(f"\n[🚀 BACKEND ACTION TRIGGERED] Lead Captured -> Name: {name} | Email: {email}\n")
    return "SUCCESS: Lead captured. Thank the user and tell them someone will reach out shortly."

tools = [mock_lead_capture]

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: NotRequired[str]

# ==========================================
# 3. LANGGRAPH NODES
# ==========================================
llm = ChatOllama(model="llama3.2", temperature=0) 

def classify_intent(state: AgentState):
    if not isinstance(state["messages"][-1], HumanMessage):
        return {"intent": state.get("intent", "Unknown")}

    user_message = str(state["messages"][-1].content).lower()
    
    # GUARDRAIL 1: State Locking. If they were already signing up, keep them in High-Intent
    if state.get("intent") == "High-Intent":
        return {"intent": "High-Intent"}

    greeting_words = ("hi", "hello", "hey", "good morning")
    high_intent_words = ("sign up", "buy", "subscribe", "purchase", "ready to start", "ready to sign up")

    if any(word in user_message for word in high_intent_words):
        return {"intent": "High-Intent"}
    elif any(word in user_message for word in greeting_words):
        return {"intent": "Greeting"}
    else:
        return {"intent": "Inquiry"}

def respond(state: AgentState):
    intent = state.get("intent", "Unknown")
    user_message = str(state["messages"][-1].content)
    
    # RAG Integration
    context_str = ""
    if intent == "Inquiry":
        docs = retriever.invoke(user_message)
        context_str = "\nKNOWLEDGE BASE FACTS:\n" + "\n".join([d.page_content for d in docs])
    
    system_prompt = f"""You are a helpful assistant for AutoStream SaaS. 
    The user's current intent is: {intent}.
    {context_str}
    
    CRITICAL RULES:
    1. If intent is 'Inquiry', answer strictly using the KNOWLEDGE BASE FACTS.
    2. If intent is 'High-Intent', you must ask for the user's Name AND Email to sign them up.
    3. Do NOT use the tool if you don't have their real name and email yet. Ask them for it!
    """

    # GUARDRAIL 2: Dynamic Tool Binding. Only give tools if High-Intent!
    if intent == "High-Intent":
        bound_llm = llm.bind_tools(tools)
    else:
        bound_llm = llm # Standard LLM, completely impossible for it to call a tool

    try:
        response = bound_llm.invoke([SystemMessage(content=system_prompt)] + state["messages"])
        return {"messages": [response]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error: {e}")]}

# ==========================================
# 4. BUILD GRAPH
# ==========================================
workflow = StateGraph(AgentState)
workflow.add_node("classifier", classify_intent)
workflow.add_node("responder", respond)
workflow.add_node("tools", ToolNode(tools)) 

workflow.add_edge(START, "classifier")
workflow.add_edge("classifier", "responder")
workflow.add_conditional_edges("responder", tools_condition)
workflow.add_edge("tools", "responder")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# ==========================================
# 5. FULL WORKFLOW TEST 
# ==========================================
if __name__ == "__main__":
    print("\nStarting Phase 4 Test: The Full Local Agent...\n")
    
    # Changed Thread ID so we don't accidentally load the bad memory from the last run!
    config: RunnableConfig = {"configurable": {"thread_id": "demo_user_final"}}

    test_inputs = [
        "Hey there!", 
        "How much is the Basic plan?",                  
        "Does it have 4k resolution?",                  
        "Actually, I'm ready to sign up for Pro!",      
        "My name is Bob.",                              
        "My email is bob@test.com"                      
    ]

    for user_input in test_inputs:
        print(f"\nUser: {user_input}")

        result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config)
        
        current_intent = result.get("intent", "Unknown")
        agent_response = result.get("messages", [])[-1].content 
        
        print(f"[Detected Intent]: {current_intent}")
        print(f"Agent: {agent_response}")
        print("-" * 50)
        time.sleep(1)