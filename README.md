-----

# 🚀 AutoStream AI: Social-to-Lead Agentic Workflow

AutoStream AI is an intelligent, locally-hosted conversational agent designed for content creators. Built entirely without paid APIs using **LangGraph** and **Ollama**, this agent seamlessly transitions from answering knowledge-base questions to actively capturing high-intent leads.

## 📑 Table of Contents

  - [Features](https://www.google.com/search?q=%23-features)
  - [Architecture & Tech Decisions](https://www.google.com/search?q=%23-architecture--tech-decisions)
  - [Project Structure](https://www.google.com/search?q=%23-project-structure)
  - [Local Setup & Installation](https://www.google.com/search?q=%23-local-setup--installation)
  - [Demo Script](https://www.google.com/search?q=%23-demo-script)
  - [WhatsApp Webhook Integration](https://www.google.com/search?q=%23-whatsapp-webhook-integration)

-----

## ✨ Features

  * **Dynamic Intent Routing:** Classifies messages on the fly into *Greeting*, *Inquiry*, or *High-Intent* to determine the appropriate workflow.
  * **100% Local RAG (Retrieval-Augmented Generation):** Uses `nomic-embed-text` and FAISS to ground answers in company facts (preventing price hallucinations).
  * **Agentic Tool Execution:** Autonomously triggers a backend `mock_lead_capture` Python function *only* when strict data requirements (Name & Email) are met.
  * **Stateful Memory:** Leverages LangGraph's `MemorySaver` to track conversation history per user, allowing contextual follow-up questions.
  * **Zero API Costs:** Runs completely offline using Meta's `Llama 3.2` model via Ollama.

-----

## 🧠 Architecture & Tech Decisions

**Why LangGraph over standard LangChain?**
Lead generation is not a linear process; it's cyclical. A user might start a checkout flow, pause to ask a question about a feature, and then resume checkout. LangGraph's cyclic state machine allows the agent to loop between the `Responder` node and `ToolNode`, retaining a "locked" state until the user's ultimate goal is achieved.

**Why Local Models?**
To avoid rate limits (HTTP 429) and API deprecations (HTTP 404), the stack was migrated from Google Gemini to local Ollama models. This ensures absolute stability for demonstrations and protects user data privacy.

-----

## 📂 Project Structure

```text
autostream-ai-agent/
│
├── main.py             # Main execution file (LangGraph setup, LLM, RAG)
├── README.md           # Project documentation
├── requirements.txt    # Python dependencies
└── .venv/              # Virtual environment
```

-----

## 💻 Local Setup & Installation

### 1\. Prerequisites

You will need Python 3.9+ and the local LLM runner [Ollama](https://www.google.com/search?q=https://ollama.com/) installed on your machine.

### 2\. Download the Models

Open your terminal and pull the necessary LLM and Embedding models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 3\. Install Python Dependencies

Create a virtual environment and install the required packages:

```bash
pip install langchain-ollama langchain-community langgraph faiss-cpu
```

### 4\. Run the Agent

```bash
python main.py
```

-----

## 🎬 Demo Script

To see the full capabilities of the agent, follow this exact dialogue in the terminal:

1.  **User:** `"How much is the Basic plan?"` *(Tests RAG retrieval)*
2.  **User:** `"Does it have 4k resolution?"` *(Tests context/memory)*
3.  **User:** `"Actually, I'm ready to sign up for Pro!"` *(Tests Intent Shift & Guardrails)*
4.  **User:** `"My name is Bob."` *(Tests state locking—agent should ask for email)*
5.  **User:** `"My email is bob@test.com"` *(Triggers the backend tool successfully\!)*

-----

## 📱 WhatsApp Webhook Integration (Concept)

To deploy this local agent to a live WhatsApp business account, the following architecture would be implemented:

1.  **Webhook Server:** A lightweight `FastAPI` application exposed to the internet via `ngrok`.
2.  **Meta Graph API:** Configured to send `POST` requests to the webhook whenever a user messages the WhatsApp business number.
3.  **Thread Management:** Map the user's WhatsApp phone number directly to LangGraph's `thread_id` to maintain unique memory for thousands of concurrent users.

**Conceptual Integration Flow:**

```python
@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    data = await request.json()
    user_phone = data["entry"][0]["changes"][0]["value"]["messages"][0]["from"]
    user_msg = data["entry"][0]["changes"][0]["value"]["messages"][0]["text"]["body"]
    
    # Use phone number as unique memory thread
    config = {"configurable": {"thread_id": user_phone}}
    
    # Pass to LangGraph
    result = app.invoke({"messages": [HumanMessage(content=user_msg)]}, config)
    agent_reply = result["messages"][-1].content
    
    # Send agent_reply back via Twilio or Meta Graph API
    send_whatsapp_message(user_phone, agent_reply)
    
    return {"status": "success"}
```
