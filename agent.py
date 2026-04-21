import os
import time
from typing import Annotated, TypedDict, NotRequired

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

# --- NEW IMPORT FOR OLLAMA ---
from langchain_ollama import ChatOllama

load_dotenv()

# --- 1. Define the Tool ---
@tool
def mock_lead_capture(name: str, email: str) -> str:
    """Use this tool ONLY when you have collected BOTH the user's Name and Email."""
    print(f"\n[🚀 BACKEND ACTION TRIGGERED] Lead Captured -> Name: {name} | Email: {email}\n")
    return "SUCCESS: Lead captured. Thank the user and tell them someone will reach out shortly."

tools = [mock_lead_capture]

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: NotRequired[str]

# --- UPDATED LLM INITIALIZATION ---
# Using llama3.1 because it natively supports tool calling!
llm = ChatOllama(
    model="llama3.2", 
    temperature=0
) 

llm_with_tools = llm.bind_tools(tools)

def classify_intent(state: AgentState):
    if not isinstance(state["messages"][-1], HumanMessage):
        return {"intent": state.get("intent", "Unknown")}

    user_message = str(state["messages"][-1].content).lower()

    greeting_words = ("hi", "hello", "hey", "good morning")
    high_intent_words = ("sign up", "buy", "subscribe", "purchase", "ready to start")

    if any(word in user_message for word in greeting_words):
        detected_intent = "Greeting"
    elif any(word in user_message for word in high_intent_words):
        detected_intent = "High-Intent"
    else:
        detected_intent = "Inquiry"

    return {"intent": detected_intent}

def respond(state: AgentState):
    intent = state.get("intent", "Unknown")
    
    system_prompt = f"""You are a helpful assistant for AutoStream SaaS. 
    The user's current intent is: {intent}.
    
    RULES:
    1. If intent is 'Greeting' or 'Inquiry', answer naturally.
    2. If intent is 'High-Intent', you MUST ask for the user's Name AND Email. 
    3. If they only give a name, ask for the email. If they only give an email, ask for the name.
    4. ONCE YOU HAVE BOTH NAME AND EMAIL, you MUST call the 'mock_lead_capture' tool. Do not call it before you have both.
    """

    try:
        response = llm_with_tools.invoke([SystemMessage(content=system_prompt)] + state["messages"])
        return {"messages": [response]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"API Error: {e}")]}

# --- 2. Build the Graph ---
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

# --- Phase 3 Test ---
if __name__ == "__main__":
    print("Starting Phase 3 Test: Tool Execution with Ollama...\n")

    config: RunnableConfig = {"configurable": {"thread_id": "test_user_3"}}

    test_inputs = [
        "I'm ready to sign up for the Pro plan!",
        "My name is Alice.",
        "It's alice@youtube.com"
    ]

    for user_input in test_inputs:
        print(f"User: {user_input}")

        result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config)
        
        current_intent = result.get("intent", "Unknown")
        agent_response = result.get("messages", [])[-1].content 
        
        print(f"[Detected Intent]: {current_intent}")
        print(f"Agent: {agent_response}\n" + "-" * 40 + "\n")
        
        time.sleep(1) # We can reduce sleep time since Ollama runs locally!