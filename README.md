# 🚀 AutoStream AI: Interactive Local Lead-Gen Agent

AutoStream AI is a stateful, interactive, and **100% locally hosted** AI agent designed for SaaS customer service and lead generation. Built using **LangGraph** and **Ollama**, this agent seamlessly answers product questions using a local Knowledge Base (RAG), rejects off-topic queries, and transitions users into a lead-capture workflow when high purchase intent is detected.

## ✨ Key Features

  * **Interactive CLI Interface:** Chat directly with the agent in a live terminal session loop.
  * **100% Local execution:** Powered by Meta's `Llama 3.2` and Nomic's `nomic-embed-text`. Zero API costs, zero rate limits, and complete data privacy.
  * **Dynamic Intent Routing:** Automatically classifies user input into *Greeting*, *Inquiry*, or *High-Intent* workflows.
  * **Local RAG (Retrieval-Augmented Generation):** Uses FAISS to ground answers strictly in company facts, preventing pricing or policy hallucinations.
  * **Agentic Tool Execution:** Autonomously triggers a backend Python function (`mock_lead_capture`) only when specific conditions (valid Name & Email) are met.
  * **Robust Guardrails:** \* **Out-of-Domain (OOD) Protection:** Politely declines to answer off-topic questions (e.g., coding, recipes) to stay in character.
      * **Tool Validation:** Prevents the LLM from hallucinating fake emails or triggering the tool prematurely.
      * **State Locking:** Uses LangGraph's `MemorySaver` to lock the user in a signup state until their details are captured.

-----

## 🧠 System Architecture

The agent operates on a **cyclical graph** architecture using LangGraph:

1.  **Classifier Node:** Analyzes the user's message and updates the state's `intent` flag.
2.  **Responder Node:** \* If `Inquiry`: Queries the FAISS vector database and answers using strictly retrieved facts.
      * If `High-Intent`: Binds the `mock_lead_capture` tool to the LLM and aggressively prompts for user details.
3.  **Tool Node:** Executes backend Python logic once valid arguments are provided by the LLM, then routes back to the Responder to thank the user.

-----

## 💻 Local Setup & Installation

### 1\. Prerequisites

You will need Python 3.9+ and the local LLM runner [Ollama](https://ollama.com/) installed on your machine.

### 2\. Download the Models

Open your terminal and pull the required LLM and Embedding models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 3\. Install Python Dependencies

Create a virtual environment (recommended) and install the required packages:

```bash
pip install langchain-ollama langchain-community langgraph faiss-cpu
```

### 4\. Run the Agent

Start the interactive live session:

```bash
python main.py
```

-----

## 🎬 Live Demo Script (Test Cases)

Once the agent is running, try typing these exact prompts to test all of its guardrails and features:

| Feature Tested | User Input | Expected Agent Behavior |
| :--- | :--- | :--- |
| **Greeting** | `Hey there!` | Responds warmly and asks how it can help. |
| **OOD Guardrail** | `Can you write me a python script for a snake game?` | Politely refuses, stating it is a customer service bot. |
| **RAG Retrieval** | `How much is the Basic plan?` | Retrieves facts and states it is $29/month. |
| **Memory / RAG** | `Does it have 4k resolution?` | Understands "it" means Basic plan, states 4K is only on Pro. |
| **Intent Shift** | `Actually, I'm ready to sign up for Pro!` | Locks state to High-Intent; asks for Name and Email. |
| **State Locking** | `My name is Alice.` | Acknowledges the name, but persists in asking for the email. |
| **Tool Execution**| `My email is alice@test.com` | Triggers `[🚀 BACKEND ACTION TRIGGERED]` and finalizes the flow. |

Type `quit`, `exit`, or `q` at any time to gracefully end the terminal session.

-----

## 🚀 Next Steps / Future Enhancements

  * **Database Integration:** Swap the `mock_lead_capture` print statement with an API call to a CRM (like HubSpot or Salesforce) or a PostgreSQL database.
  * **Live Knowledge Base:** Replace the hardcoded `kb_data` array with a document loader that ingests real company PDFs or web scrapes the company FAQ page.
  * **Web UI / Chatbot API:** Wrap the LangGraph execution block in a FastAPI endpoint to connect it to a React frontend or a WhatsApp Twilio integration.

-----


