# AI Engineering Playground

A personal collection of hands-on experiments, examples, and proof-of-concepts built while learning and exploring modern AI engineering frameworks and tooling.

The repository is organized by technology, with each module focusing on a specific framework or area of the LLM ecosystem. The goal is to build practical understanding through small, focused examples before combining these concepts into larger AI applications.

> **Note:** This is a learning repository. The examples are intentionally small and self-contained, focusing on understanding concepts rather than building production-ready applications.

---

## Repository Structure

```
.
├── src/
│   ├── langchain/
│   │   ├── prompts/
│   │   ├── messages/
│   │   ├── chains/
│   │   ├── conversation_history/
│   │   ├── runnables/
│   │   └── ...
│   │
│   ├── deepeval/          # Planned
│   ├── langgraph/         # Planned
│   ├── rag/               # Planned
│   ├── agents/            # Planned
│   └── mcp/               # Planned
│
├── pyproject.toml
├── uv.lock
├── .python-version
└── README.md
```

---

## Current Modules

### LangChain

Currently contains examples covering:

- Prompt Templates
- Chat Prompt Templates
- LangChain Messages
- MessagesPlaceholder
- Conversation History
- Prompt Partial Variables
- Runnable Interface
- Sequential Chains
- Parallel Chains
- Multi-Prompt Workflows
- Message Arrays
- Prompt Engineering Basics

---

## Planned Modules

The repository will continue to expand with practical examples covering:

- DeepEval
- LangGraph
- Retrieval-Augmented Generation (RAG)
- AI Agents
- Model Context Protocol (MCP)
- Vector Databases
- Embeddings
- Tool Calling
- Structured Output
- Streaming
- Memory
- Observability
- AI Evaluation
- Guardrails
- AI Testing

---

## Prerequisites

- Python 3.12+
- uv
- Git
- Ollama (for local LLMs)

---

## Setup

Clone the repository

```bash
git clone <repository-url>
cd langchain-project
```

Create a virtual environment

```bash
uv venv
```

Activate it

### Windows (PowerShell)

```powershell
.venv\Scripts\Activate.ps1
```

Install project dependencies

```bash
uv sync
```

---

## Environment Variables

Create a `.env` file in the project root.

Example:

```text
OPENAI_API_KEY=your_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true

OLLAMA_KEEP_ALIVE=24h
```

Depending on the module being executed, additional environment variables may be required.

---

## Running Examples

Run any example using `uv run`.

Example:

```bash
uv run src/langchain/simple_prompt.py
```

or

```bash
uv run src/langchain/multiple_chains.py
```

---

## Technologies

Current

- Python
- LangChain
- uv
- Ollama
- OpenAI
- LangSmith
- python-dotenv

Planned

- DeepEval
- LangGraph
- ChromaDB / FAISS
- MCP
- OpenTelemetry
- Playwright
- Hugging Face
- LiteLLM

---

## Learning Objectives

This repository is used to gain hands-on experience with:

- Prompt Engineering
- LLM Application Development
- Chain Composition
- Runnable Interfaces
- Conversation Management
- Local LLM Integration
- AI Evaluation
- Retrieval-Augmented Generation
- Agentic AI
- Workflow Orchestration
- Observability and Tracing
- Testing AI Systems
- Production AI Engineering Patterns

---

## Roadmap

- ✅ LangChain fundamentals
- ⏳ DeepEval experiments
- ⏳ LangGraph workflows
- ⏳ RAG implementations
- ⏳ AI Agents
- ⏳ MCP integrations
- ⏳ Evaluation frameworks
- ⏳ Production-ready AI patterns

---

## License

This repository is intended for educational and learning purposes.