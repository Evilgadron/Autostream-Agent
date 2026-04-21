
# 🚀 Social-to-Lead Agentic Workflow (AutoStream AI)

An intelligent, 100% local Agentic workflow built for **AutoStream** (a fictional SaaS for automated video editing). This agent uses **LangGraph** to manage state, **RAG** (Retrieval-Augmented Generation) to answer product questions, and **Ollama** to provide a private, local-first LLM experience.

## 🌟 Key Features
- **Intent Classification:** Automatically categorizes user messages into Greeting, Inquiry, or High-Intent.
- **Local RAG Pipeline:** Uses `nomic-embed-text` and FAISS to provide accurate pricing and policy information.
- **Autonomous Tool Execution:** Triggers a `mock_lead_capture` tool only when a high-intent user provides both Name and Email.
- **Stateful Memory:** Retains conversation context across multiple turns using LangGraph's `MemorySaver`.
- **100% Local:** Powered by Ollama (`llama3.2`), ensuring data privacy and zero API costs.

---

## 🛠️ Architecture Explanation

For this project, I chose **LangGraph** over standard linear chains. Standard LLM chains often struggle with "loops"—situations where an agent needs to ask follow-up questions before executing a tool. LangGraph treats the conversation as a **State Machine (Directed Acyclic Graph)**. 

### Why LangGraph?
LangGraph allows for "Cycles." If a user provides a name but forgets an email, the graph can loop back to the `responder` node until the `tools_condition` is finally met. This provides much higher reliability for lead qualification than a simple prompt-based approach.

### State Management
State is managed via a `TypedDict` called `AgentState`, which tracks the list of messages and the current detected intent. To ensure memory persists across turns, I implemented a `MemorySaver` checkpointer. By using a `thread_id` in the `RunnableConfig`, the agent can distinguish between different users and remember their specific details (like their name) from five turns ago.

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.9+**
- **Ollama** installed (Download from [ollama.com](https://ollama.com))

### 2. Setup Ollama Models
The agent requires a text-generation model and an embedding model. Run:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 3. Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 4. Running the Agent
```bash
python main.py
```

---

## 📱 WhatsApp Deployment Strategy
To deploy this agent on WhatsApp, I would follow these steps:
1.  **Twilio / WhatsApp Business API:** Use Twilio as the gateway to the WhatsApp API.
2.  **Webhook Integration:** Set up a FastAPI or Flask endpoint. When a user sends a message, WhatsApp triggers a POST request (Webhook) to this endpoint.
3.  **State Persistence:** In the backend, the user's WhatsApp number (e.g., `+123456789`) would serve as the `thread_id` for LangGraph, ensuring the agent remembers that specific user's conversation history.
4.  **Response:** The agent processes the message, and the response is sent back to the user via the Twilio Messaging API.

---

## 📽️ Demo Video
1. **RAG Test:** The agent accurately identifies that the Basic plan costs $29.
2. **Memory Test:** The agent remembers the user's name across several messages.
3. **Intent Detection:** The agent switches from "Inquiry" to "High-Intent" when I say I'm ready to sign up.
4. **Tool Trigger:** The terminal prints the `[🚀 BACKEND ACTION TRIGGERED]` message once the email is provided.

---
