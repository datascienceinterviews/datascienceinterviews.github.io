---
title: LangChain & LangGraph Cheat Sheet
description: Comprehensive LangChain and LangGraph reference covering setup, prompts, LCEL, parsers, memory, agents, RAG, embeddings, graphs, routing, serving, deployment, monitoring, and best practices with ASCII diagrams.
---

# LangChain & LangGraph Cheat Sheet

[TOC]

This cheat sheet provides a deep, end-to-end reference for LangChain and LangGraph: core components, LCEL, memory, agents, RAG, embeddings, graph construction, routing, serving/deployment, and monitoring. Each concept is shown with a concise explanation, ASCII/Unicode box-drawing diagram, practical code, and actionable tips.

??? note "**LangChain / LangGraph Visual Cheatsheets**"

    ![LangChain Cheat Sheet 1](../assets/img/LangChain-Cheatsheet-1.webp)
    
    ![LangChain Cheat Sheet 2](../assets/img/LangChain-Cheatsheet-2.webp)
    
    ![LangChain Cheat Sheet 3](../assets/img/LangChain-Cheatsheet-3.webp)

## Quick Start (LCEL minimal chain)
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are concise."),
    ("user", "Question: {question}")
])
chain = prompt | llm | StrOutputParser()

print(chain.invoke({"question": "What is LangChain?"}))
```

## Important Links
- LangChain Python Docs: https://python.langchain.com
- LangGraph Docs: https://langchain-ai.github.io/langgraph/
- LangServe Docs: https://python.langchain.com/docs/langserve
- LangSmith Docs: https://docs.smith.langchain.com
- LCEL Guide: https://python.langchain.com/docs/expression_language/

## Getting Started
**Brief:** Install core packages, set API keys, and verify environment.

**Diagram:**
```text
┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│  Install     │→→→│  Configure   │→→→│  Validate    │
│  langchain   │    │  API Keys   │    │  imports    │
└──────────────┘    └──────┬──────┘    └──────┬──────┘
                            │                 │
                            ↓                 ↓
                   ┌────────────────┐   ┌────────────┐
                   │ OPENAI_API_KEY │   │ python -c  │
                   │ HUGGINGFACE_*  │   │  "import   │
                   └────────────────┘   │  langchain"│
                                        └────────────┘
```

**Code:**
```bash
pip install "langchain>=0.3" "langgraph>=0.2" langchain-openai langchain-community langchain-core
# Optional extras: vector stores, parsers, serving
pip install faiss-cpu chromadb pinecone-client weaviate-client pydantic openai fastapi uvicorn redis
```
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
# Optional: LangSmith for tracing
os.environ["LANGCHAIN_API_KEY"] = "ls_..."
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
```

**Tips:**
- Keep secrets in env vars or `.env` + `python-dotenv`.
- Pin versions for reproducibility (`pip freeze > requirements.txt`).
- Verify GPU/CPU availability before large embeddings.

## Core LangChain Components
**Brief:** Models (LLMs vs Chat), Prompts, Output Parsers, Chains.

**Diagram:**
```text
┌─────────┐   ┌──────────┐   ┌──────────────┐   ┌─────────┐
│ Input   │ → │ Prompt   │ → │ Model (LLM/  │ → │ Parser  │
│ (user)  │   │ Template │   │ ChatModel)   │   │ (struct)│
└─────────┘   └──────────┘   └──────┬───────┘   └────┬────┘
                                     │              │
                                     ↓              ↓
                                 ┌────────┐    ┌──────────┐
                                 │ Chain  │ →  │ Output   │
                                 └────────┘    └──────────┘
```

**Code:**
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are concise."),
    ("user", "Question: {question}")
])
chain = prompt | llm | StrOutputParser()
print(chain.invoke({"question": "What is LangChain?"}))
```

**Tips:**
- Use `temperature=0` for deterministic outputs; higher for creativity.
- Always parse outputs for downstream safety (JSON, Pydantic, etc.).
- Prefer Chat models for function/tool calling.

## Prompt Templates (String, Chat, Few-Shot)
**Brief:** Parameterized prompts to control style and context.

**Diagram:**
```text
┌─────────────┐   variables   ┌─────────────────┐
│ Base Prompt │ ───────────→  │ Rendered Prompt │
└─────┬───────┘               └────────┬────────┘
      │ few-shot examples               │ to model
      ↓                                 ↓
┌─────────────┐                 ┌───────────────┐
│ Example 1   │                 │ Chat Messages │
├─────────────┤                 └───────────────┘
│ Example 2   │
└─────────────┘
```

**Code:**
```python
from langchain_core.prompts import (PromptTemplate, ChatPromptTemplate,
                                    FewShotPromptTemplate)

string_prompt = PromptTemplate.from_template("Translate to French: {text}")
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a translator."),
    ("user", "Translate: {text}")
])
examples = [
    {"text": "hello", "translation": "bonjour"},
    {"text": "good night", "translation": "bonne nuit"},
]
example_prompt = PromptTemplate.from_template("Input: {text}\nOutput: {translation}")
few_shot = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Use examples to guide style.",
    suffix="Input: {text}\nOutput:",
    input_variables=["text"],
)
print(few_shot.format(text="thank you"))
```

**Tips:**
- Keep examples short; match target style.
- Use placeholders consistently; validate with `prompt.format()`.
- For chat, keep system message focused; add guardrails.

## LCEL (LangChain Expression Language)
**Brief:** Compose chains with `|`, parallel branches, passthrough, streaming, async.

**Diagram (LCEL Pipe Flow):**
```text
Input
  │
  ├─→ ┌──────────┐ → ┌────────────┐ → ┌──────────────┐ → Output
  │   │ Transform │   │ Transform  │   │ Transform    │
  │   └──────────┘   └────────────┘   └──────────────┘
  │
  └─→ ┌───────────────┐
      │ RunnableParallel│ (fan-out, then merge)
      └───────────────┘
```

**Code:**
```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize and extract keywords."),
    ("user", "{text}")
])
branch = RunnableParallel(
    summary=prompt | llm | StrOutputParser(),
    keywords=prompt | llm | StrOutputParser(),
)
chain = RunnablePassthrough.assign(text=lambda x: x["text"]) | branch
result = chain.invoke({"text": "LangChain simplifies LLM orchestration."})
print(result)
```

**Tips:**
- Use `assign` to enrich inputs without losing originals.
- `astream()`/`astream_events()` for streaming tokens/events.
- Compose sync/async seamlessly; prefer async for I/O-heavy pipelines.

## Output Parsers (Pydantic, JSON, Structured)
**Brief:** Enforce structured outputs for safety and determinism.

**Diagram:**
```text
┌───────────┐   ┌──────────────┐   ┌────────────────┐
│ LLM Text  │ → │ OutputParser │ → │ Typed Object   │
└───────────┘   └──────┬───────┘   └────────┬───────┘
                       │ jsonschema         │ pydantic
                       ↓                    ↓
                ┌─────────────┐     ┌──────────────┐
                │ Validation  │     │ raise errors │
                └─────────────┘     └──────────────┘
```

**Code:**
```python
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

class Answer(BaseModel):
    summary: str = Field(..., description="Brief answer")
    sources: list[str]

parser = PydanticOutputParser(pydantic_object=Answer)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Return JSON matching the schema."),
    ("user", "Question: {question}\n{format_instructions}")
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser
print(chain.invoke({"question": "What is LangGraph?"}))
```

**Tips:**
- Use `OutputFixingParser` to auto-correct near-misses.
- Prefer `StructuredOutputParser`/`PydanticOutputParser` for reliability.
- Validate early before persisting to DBs.

## Memory Systems
**Brief:** Store conversational context: buffer, summary, windowed, vector, entity-aware.

**Diagram (Memory Types Comparison):**
```text
┌───────────────┬───────────────────────┬──────────────────────────┐
│ Type          │ Pros                  │ Cons                     │
├───────────────┼───────────────────────┼──────────────────────────┤
│ Buffer        │ Exact history         │ Grows unbounded          │
│ BufferWindow  │ Recent k messages     │ Loses older context      │
│ Summary       │ Compressed narrative  │ Possible detail loss     │
│ VectorStore   │ Semantic recall       │ Needs embeddings/store   │
│ EntityMemory  │ Tracks entities slots │ Setup complexity         │
└───────────────┴───────────────────────┴──────────────────────────┘
```

**Code:**
```python
from langchain.memory import (ConversationBufferMemory,
    ConversationBufferWindowMemory, ConversationSummaryMemory,
    ConversationEntityMemory)
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

buffer = ConversationBufferMemory(return_messages=True)
window = ConversationBufferWindowMemory(k=3, return_messages=True)
summary = ConversationSummaryMemory(llm=llm, return_messages=True)
entity = ConversationEntityMemory(llm=llm)

# Vector store memory
embedding = OpenAIEmbeddings()
vs = FAISS.from_texts(["Hello world"], embedding)
```

**Tips:**
- Pick memory based on cost vs fidelity: window for short chats; summary for long.
- Vector memory helps retrieve semantic context; tune chunk size/overlap.
- Clear memory per session to avoid leakage across users.

## Agents & Tools
**Brief:** Agents choose tools dynamically; tools expose functions.

**Diagram (Agent Decision Loop):**
```text
                ┌─────────────────┐
                │   Start Agent   │
                └────────┬────────┘
                         ↓
                ┌────────────────────────────────┐
                │      OBSERVE                    │
                │  - User query                   │
                │  - Previous tool outputs        │
                │  - Current state                │
                └────────┬───────────────────────┘
                         ↓
                ┌────────────────────────────────┐
                │   THINK (LLM Reasoning)        │
                │  - Analyze observation         │
                │  - Review available tools      │
                │  - Plan next action            │
                └────────┬───────────────────────┘
                         ↓
                ┌────────────────────┐
                │      DECIDE        │
                └────────┬───────────┘
                         │
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Use Tool A  │  │ Use Tool B  │  │ Final Answer│
└─────────────┘  └─────────────┘  └─────────────┘
                         ↓
                ┌────────────────────┐
                │       ACT          │
                └────────────────────┘
```

**Diagram (Agent with Multiple Tools):**
```text
           ┌──────────────┐
           │ Agent Brain  │
           └──────┬───────┘
                  ↓
    ┌───────┬───────────┬───────────┬───────────┐
    ↓       ↓           ↓           ↓           ↓
┌──────┐ ┌───────┐  ┌────────┐  ┌────────┐  ┌─────────┐
│Search│ │Calc    │ │Weather │ │Custom  │ │Feedback  │
│API   │ │Tool    │ │Tool    │ │Tool    │ │Loop      │
└──┬───┘ └──┬─────┘  └──┬────┘  └──┬────┘  └────┬─────┘
   │        │           │         │            │
   └────────┴───────────┴─────────┴────────────┘
                  ↑ aggregated observation
```

**Diagram (Agent Type Selection Tree):**
```text
               ┌───────────────────────┐
               │ Need structured tools?│
               └───────────┬───────────┘
                           │Yes
                           ↓
                   ┌─────────────┐
                   │ Tool-Calling│ (OpenAI-style)
                   └────┬────────┘
                        │No
                        ↓
          ┌────────────────────────┐
          │ Dialogue heavy?        │
          └─────────┬──────────────┘
                    │Yes            │No
                    ↓               ↓
         ┌────────────────┐   ┌─────────────────────┐
         │ Conversational │   │ Zero-Shot ReAct     │
         │ Agent          │   │ (plan & act inline) │
         └────────────────┘   └─────────┬───────────┘
                                         │ Need long plans?
                                         ↓
                                 ┌──────────────┐
                                 │ Plan-and-Exec│
                                 └──────────────┘
```

**Code:**
```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

llm_tools = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Use tools when helpful."),
    ("user", "{input}")
])
agent = create_tool_calling_agent(llm_tools, [multiply], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[multiply], verbose=True)
print(agent_executor.invoke({"input": "What is 12*9?"}))
```

**Tips:**
- Keep tool signatures small, well-described docstrings.
- Return structured JSON to simplify parsing.
- Add guardrails (e.g., tool whitelists, max iterations).

## Agent Execution (loops, errors)
**Brief:** `AgentExecutor` orchestrates reasoning, tool calls, retries.

**Diagram:**
```text
┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────┐
│ User In  │→→│ Agent Executor│→→│ Tool Calls    │→→│ Final Out │
└────┬─────┘   └──────┬───────┘   └──────┬───────┘   └────┬─────┘
         │ retry/backoff         │ errors    │ streaming tokens │
         ↓                       ↓           ↓                 ↓
    Logs/Traces           Error Handler  Partial Results   Return
```

**Code:**
```python
result = agent_executor.invoke({"input": "Plan weekend: check weather in NYC and suggest indoor/outdoor."})
# For streaming thoughts
for event in agent_executor.astream_events({"input": "..."}):
    print(event)
```

**Tips:**
- Set `handle_parsing_errors=True` or custom handler for robust runs.
- Cap `max_iterations`; log intermediate steps.
- Include tool observation snippets in prompts to avoid loops.

## RAG - Retrieval Augmented Generation
**Brief:** Retrieve relevant chunks then generate grounded answers.

**Diagram (RAG Full Pipeline):**
```text
┌────────────┐   ┌───────────┐   ┌──────────┐   ┌────────────┐   ┌──────────┐
│ Document   │ → │ Loaders   │ → │ Splitter │ → │ Embeddings │ → │ Vector DB│
│ Store/FS   │   │ (PDF/URL) │   │ chunks   │   │ (OpenAI/   │   │ (FAISS/  │
└─────┬──────┘   └────┬──────┘   └────┬─────┘   │ HF/Cohere) │   │ Chroma…) │
      │               │             │          └────┬──────┘   └────┬─────┘
      │               │             │               │              │
      │               │             │               ↓              ↓
      │               │             │        ┌────────────┐  ┌────────────┐
      │ Query         │             │        │ Similarity  │  │ Retrieved  │
      └───────────────┴─────────────┴──────→ │ Search      │→ │ Chunks     │
                                             └────┬───────┘  └────┬──────┘
                                                  ↓              ↓
                                            ┌──────────────────────────┐
                                            │ LLM (prompt with context)│
                                            └──────────────────────────┘
```

**Diagram (RAG Chain Types Table):**
```text
┌────────────┬───────────────┬────────────────────┬──────────────────────┐
│ Strategy   │ How           │ Pros               │ Cons                 │
├────────────┼───────────────┼────────────────────┼──────────────────────┤
│ Stuff      │ Concatenate   │ Simple, fast       │ Context length bound │
│ Map-Reduce │ Map chunks -> │ Scales, summaries  │ More LLM calls       │
│            │ partial, then │                    │                      │
│            │ reduce        │                    │                      │
│ Refine     │ Iterative add │ Keeps detail       │ Sequential latency   │
│ Map-Rerank │ Score each    │ Better precision   │ Costly reranking     │
└────────────┴───────────────┴────────────────────┴──────────────────────┘
```

**Code:**
```python
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

loader = WebBaseLoader("https://python.langchain.com")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
chunks = splitter.split_documents(docs)
emb = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, emb)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_reduce",  # or "stuff", "refine", "map_rerank"
    retriever=retriever,
)
print(qa_chain.invoke({"query": "How does LCEL work?"}))
```

**Tips:**
- Tune `chunk_size` to ~200-1000 tokens; overlap ~10-20%.
- Choose chain type per corpus size: `stuff` for small, `map_reduce` for large.
- Add citations by returning source metadata in prompt.

## Vector Stores (FAISS, Chroma, Pinecone, Weaviate)
**Brief:** Persistent similarity search backends.

**Diagram:**
```text
┌────────────┐   ┌─────────────┐   ┌─────────────┐
│ Text/Chunk │→→│ Embeddings   │→→│ Vector Store │
└────┬───────┘   └──────┬──────┘   └──────┬──────┘
     │                  │               │
     ↓                  ↓               ↓
 Query Text → embed → similarity search → Top-k IDs → Fetch docs
```

**Code:**
```python
# FAISS (local)
faiss_store = FAISS.from_documents(chunks, emb)
faiss_store.save_local("faiss_index")
faiss_loaded = FAISS.load_local("faiss_index", emb, allow_dangerous_deserialization=True)

# Chroma (local serverless)
from langchain_community.vectorstores import Chroma
chroma = Chroma.from_documents(chunks, emb, collection_name="docs")

# Pinecone (managed)
import pinecone
pinecone.Pinecone(api_key="...")
from langchain.vectorstores import Pinecone as PineconeVS
pinecone_index = pinecone.Index("langchain-demo")
pinecone_vs = PineconeVS(index=pinecone_index, embedding_function=emb.embed_query)

# Weaviate (managed/self-hosted)
import weaviate
client = weaviate.Client("https://xyz.weaviate.network", auth_client_secret=weaviate.AuthApiKey("..."))
```

**Tips:**
- Pick HNSW (Chroma/Weaviate) for fast recall; IVF/Flat in FAISS for precision.
- Normalize vectors for cosine similarity when required.
- Persist indexes; align embedding model at query and ingest time.

## Document Processing (Loaders, Splitters)
**Brief:** Load diverse sources and split text for retrieval.

**Diagram (Text Splitter Strategies):**
```text
┌──────────────┬───────────────────────────┬──────────────────────────┐
│ Strategy     │ How                       │ Best For                 │
├──────────────┼───────────────────────────┼──────────────────────────┤
│ Character    │ Fixed chars + overlap     │ Clean text               │
│ Recursive    │ Fallback by delimiters    │ Mixed formats            │
│ Token-based  │ Token counts (tiktoken)   │ Token budgets            │
│ Semantic     │ Embedding-based merge     │ Coherent chunks          │
└──────────────┴───────────────────────────┴──────────────────────────┘
```

**Code:**
```python
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import (CharacterTextSplitter,
    RecursiveCharacterTextSplitter, TokenTextSplitter)

pdf_docs = PyPDFLoader("file.pdf").load()
csv_docs = CSVLoader("data.csv").load()

rec_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
chunks = rec_splitter.split_documents(pdf_docs + csv_docs)
```

**Tips:**
- Strip headers/footers before splitting PDFs when possible.
- Use `TokenTextSplitter` for strict token budgets.
- Preserve metadata (page, URL) for citations.

## Embeddings (OpenAI, HF, Cohere, local)
**Brief:** Convert text to vectors for similarity search.

**Diagram:**
```text
┌──────────┐   ┌──────────────┐   ┌──────────────┐
│ Raw Text │→→│ Embed Model   │→→│ Vector (dims) │
└──────────┘   └──────┬───────┘   └──────┬───────┘
                  │ norm           │ store/reuse
                  ↓                ↓
              Cache/DB        Retrieval pipelines
```

**Code:**
```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings, CohereEmbeddings

openai_emb = OpenAIEmbeddings(model="text-embedding-3-small")
hf_emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
cohere_emb = CohereEmbeddings(model="embed-english-light-v3.0", cohere_api_key="...")
```

**Tips:**
- Match embedding model language/domain to corpus.
- Batch embeddings to reduce latency; cache results.
- For local privacy, prefer HF or GGUF-based models.

## LangGraph Basics (state)
**Brief:** Build graphs where nodes are steps; state carried via typed dicts/reducers.

**Diagram (StateGraph Execution):**
```text
┌────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐
│ START  │ → │ Node A   │ → │ Node B   │ → │ END    │
└────────┘   └────┬─────┘   └────┬─────┘   └────────┘
                  │            │
                  └─────→──────┘ (conditional edge)
```

**Diagram (State Reducer Behavior):**
```text
With reducer (operator.add):           Without reducer:
┌──────────┐   ┌──────────┐           ┌──────────┐   ┌──────────┐
│ state=1  │ → │ add 2    │ = 3      │ state=1  │ → │ set 2    │ = 2
└──────────┘   └──────────┘           └──────────┘   └──────────┘
```

**Code:**
```python
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, END

class GraphState(TypedDict):
    message: str
    steps: Annotated[list[str], add]  # reducer concatenates lists

graph = StateGraph(GraphState)

def start(state: GraphState):
    return {"message": state["message"], "steps": ["start"]}

def finish(state: GraphState):
    return {"message": state["message"], "steps": state["steps"] + ["finish"]}

graph.add_node("start", start)
graph.add_node("finish", finish)
graph.add_edge("start", "finish")
graph.set_entry_point("start")
graph.set_finish_point("finish")
compiled = graph.compile()
print(compiled.invoke({"message": "hi", "steps": []}))
```

**Tips:**
- Use reducers (e.g., `operator.add`) for accumulating lists/counters safely.
- TypedDict enforces state shape; fail fast on missing keys.
- Prefer pure functions for nodes; keep side-effects minimal.

## LangGraph Construction (nodes/edges)
**Brief:** Build graphs with nodes, edges, conditionals.

**Code:**
```python
from langgraph.graph import StateGraph, END

g = StateGraph(GraphState)

g.add_node("decide", lambda s: {"route": "a" if "math" in s["message"] else "b"})

g.add_node("tool_a", lambda s: {"result": "used A"})
g.add_node("tool_b", lambda s: {"result": "used B"})

# Conditional routing
from langgraph.graph import add_conditional_edges

add_conditional_edges(
    g,
    source="decide",
    path_map={"a": "tool_a", "b": "tool_b"},
)

g.add_edge("tool_a", END)
g.add_edge("tool_b", END)

g.set_entry_point("decide")
compiled = g.compile()
print(compiled.invoke({"message": "math question", "steps": []}))
```

**Tips:**
- Always set entry + finish points.
- Use `add_conditional_edges` for clean branching logic.
- Keep node names descriptive; log transitions for debugging.

## LangGraph Execution (invoke, stream, checkpoint)
**Brief:** Run graphs sync/async with streaming and persistence.

**Diagram:**
```text
┌────────┐   invoke()   ┌──────────┐   stream tokens   ┌────────────┐
│ Client │────────────→│ Graph    │────────────────→  │ Responses  │
└────────┘              └──────────┘                  └────────────┘
       │ checkpoint
       └────────────→ storage (Redis/S3/DB)
```

**Code:**
```python
compiled = graph.compile(checkpointer=None)  # or Redis/S3 checkpointer

# Single call
compiled.invoke({"message": "hi", "steps": []})

# Streaming
for event in compiled.astream_events({"message": "hi", "steps": []}):
    print(event)

# Checkpointing with Redis (resume later)
from langgraph.checkpoint.redis import RedisCheckpointSaver
import redis

r = redis.Redis(host="localhost", port=6379, db=0)
checkpointer = RedisCheckpointSaver(r)
compiled_ckpt = graph.compile(checkpointer=checkpointer)
run = compiled_ckpt.invoke({"message": "hi", "steps": []})
# ... later
compiled_ckpt.resume(run["checkpoint_id"])
```

**Tips:**
- Use a checkpointer (e.g., Redis) for resumable flows.
- Prefer streaming for chat UX; buffer for batch jobs.
- Persist state for human handoffs or crash recovery.

## Conditional Routing
**Brief:** Route based on state values or model decisions.

**Diagram:**
```text

# Simple API key guard
API_KEY = "changeme"
@app.middleware("http")
async def auth(request: Request, call_next):
        if request.headers.get("x-api-key") != API_KEY:
                raise HTTPException(status_code=401, detail="Unauthorized")
        return await call_next(request)
┌────────┐   ┌────────────┐   ┌──────────────┐
│ Input  │→→│ Router Node │→→│ Branch A      │
└────────┘   └────┬───────┘   └─────┬────────┘
                  │                │

**Test/Call with curl:**
```bash
curl -X POST "http://localhost:8000/chain/invoke" \
    -H "x-api-key: changeme" \
    -H "Content-Type: application/json" \
    -d '{"input": {"question": "Hello"}}'
```
                  └──────→─────────┘ Branch B
```

**Code:**
```python
def router(state):
    if "finance" in state["message"]:
        return "finance"
    return "general"

add_conditional_edges(g, "decide", {"finance": "tool_b", "general": "tool_a"})
```

**Tips:**
- Keep routing functions pure and deterministic when possible.
- For LLM-based routing, constrain outputs (JSON labels) and validate.
- Add default fallbacks to avoid dead ends.

## Multi-Agent Patterns
**Brief:** Supervisor coordinates specialist agents.

**Diagram (Multi-Agent Supervisor):**
```text
┌──────┐   ┌─────────────┐   ┌──────────────┐
│User  │→→│ Supervisor   │→→│ Task Decompose│
└──┬───┘   └──────┬──────┘   └──────┬───────┘
   │              │                │
   ↓              ↓                ↓
┌────────┐  ┌──────────┐    ┌────────────┐    ┌─────────────┐
│Research│  │Writer    │    │FactChecker │    │CodeRunner   │
└────┬───┘  └────┬─────┘    └────┬──────┘    └────┬────────┘
     │          │               │                 │
     └──────────┴─────→─────────┴────→────────────┘
                    Aggregation → Final Answer
```

**Code:**
```python
# Pseudocode skeleton
supervisor = compiled  # a LangGraph coordinating agents
# Each specialist is a tool-calling chain; supervisor routes tasks
```

**Tips:**
- Give each agent narrow scope + tools; supervisor merges.
- Prevent loops with max hops/iterations.
- Log per-agent traces for debugging.

## Human-in-the-Loop
**Brief:** Interrupt, review, then resume with checkpoints.

**Diagram (Human-in-the-Loop Flow):**
```text
┌─────────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐
│ Execute     │→→│ Checkpoint │→→│ Interrupt  │→→│ Human OK? │
└─────┬───────┘   └────┬──────┘   └────┬──────┘   └────┬──────┘
      │                │             │             │Yes │No
      ↓                ↓             ↓             ↓    ↓
   Resume ←────────── Save        Review      Reject/Amend
```

**Code:**
```python
# Use a checkpointer; pause on specific node
state = compiled.invoke(...)
# Later, resume with stored checkpoint id
compiled.resume(checkpoint_id="abc123")
```

**Tips:**
- Define explicit pause points (e.g., before external actions).
- Store human feedback in state for auditability.
- Timebox approvals to avoid stale sessions.

## Serving & Deployment (LangServe + FastAPI)
**Brief:** Expose chains/graphs as REST endpoints.

**Diagram (LangServe Deployment):**
```text
Client → API Gateway → LangServe (FastAPI) → Chain/Graph → Response
```

**Code (LangServe):**
```python
# app.py
from fastapi import FastAPI
from langserve import add_routes
from my_chains import chain, graph

app = FastAPI()
add_routes(app, chain, path="/chain")
add_routes(app, graph, path="/graph")

# Run
# uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Tips:**
- Validate request schemas; limit concurrency via worker settings.
- Use `uvicorn --workers` for CPU-bound or async I/O.
- Add auth (API keys/JWT) at gateway or FastAPI middleware.

## Production Deployment (Docker, scaling)
**Brief:** Containerize, scale horizontally, add caching and observability.

**Diagram (Production Architecture):**
```text
┌──────────────┐   ┌───────────────┐   ┌───────────┐   ┌──────────┐
│ Clients      │→→│ Load Balancer │→→│ App Pods  │→→│ Vector DB │
└──────────────┘   └──────┬────────┘   └────┬──────┘   └────┬─────┘
                          │                 │             │
                          ↓                 ↓             ↓
                     ┌────────┐       ┌─────────┐   ┌─────────┐
                     │ Redis  │       │ Postgres│   │ Metrics │
                     │ Cache  │       │ /S3     │   │+Tracing │
                     └────────┘       └─────────┘   └─────────┘
```

**Code (Dockerfile snippet):**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Tips:**
- Externalize secrets via env vars/secret managers.
- Use autoscaling on CPU/RAM/QPS; warm LLM connections.
- Add health probes (`/healthz`) and readiness checks.

**Docker Compose (app + chroma + prometheus minimal):**
```yaml
services:
    app:
        build: .
        environment:
            - OPENAI_API_KEY=${OPENAI_API_KEY}
        ports: ["8000:8000"]
        depends_on: [chroma]
    chroma:
        image: ghcr.io/chroma-core/chroma:latest
        ports: ["8001:8000"]
    prometheus:
        image: prom/prometheus:latest
        ports: ["9090:9090"]
```

## LangSmith Monitoring
**Brief:** Trace, debug, and evaluate chains/agents.

**Diagram:**
```text
┌──────────┐   ┌──────────┐   ┌──────────────┐   ┌────────────┐
│ Chains   │→→│ Tracer    │→→│ LangSmith UI  │→→│ Insights    │
└────┬─────┘   └────┬─────┘   └────┬─────────┘   └────┬───────┘
         │ logs/errors       │ spans         │ filters       │ actions
         ↓                   ↓               ↓              ↓
     Storage          Timing/Cost     Compare runs   Prompt fixes
```

**Code:**
```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "my-app"
# Traces auto-captured when using langchain core runtimes
```

**Tips:**
- Tag runs with metadata (user id, version) for filtering.
- Use datasets + evals to compare prompt/model changes.
- Inspect tool call errors to tighten parsing.

## Caching & Optimization
**Brief:** Reduce latency/cost via caching and prompt/model choices.

**Diagram:**
```text
┌──────────┐   ┌──────────┐   ┌──────────┐
│ Request  │→→│ Cache?    │→→│ Response  │
└────┬─────┘   └────┬─────┘   └────┬─────┘
         │ miss         │ hit         │
         ↓              ↓             ↓
     Call LLM   Return cached   Store result
```

**Code:**
```python
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache
set_llm_cache(SQLiteCache(database_path=".langchain_cache.sqlite"))
```

**Tips:**
- Cache deterministic calls (temperature 0) keyed on prompt+input.
- Use shorter prompts, smaller models (`gpt-4o-mini`, `gpt-3.5-turbo`) for bulk.
- Batch embeddings; reuse vector store across sessions.

**Cost/Latency quick picks:**
```text
┌────────────┬─────────────────────────────┬────────────────────┐
│ Use Case   │ Model/Setting               │ Why                │
├────────────┼─────────────────────────────┼────────────────────┤
│ Cheap bulk │ gpt-3.5-turbo, temp 0       │ Low cost, fast     │
│ Quality    │ gpt-4o-mini, temp 0.2       │ Balance quality    │
│ Max quality│ gpt-4o, temp 0-0.3          │ Best reasoning     │
│ RAG ingest │ chunk 500-800, overlap 10%  │ Good recall/size   │
│ Caching    │ Redis/SQLite, key prompt+in │ Cut repeat costs   │
└────────────┴─────────────────────────────┴────────────────────┘
```

**Testing prompts/parsers (pytest sketch):**
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def test_prompt_format():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Be short."),
        ("user", "{q}")
    ])
    rendered = prompt.format(q="hi")
    assert "hi" in rendered

def test_parser_deterministic(monkeypatch):
    parser = StrOutputParser()
    # Mock LLM call
    class FakeLLM:
        def invoke(self, *_args, **_kwargs):
            return "ok"
    chain = FakeLLM() | parser  # type: ignore
    assert chain.invoke({}) == "ok"
```

## Advanced Patterns (Reflection, ReAct, tools in graphs)
**Brief:** Improve reliability with self-critique, tool use, and ReAct.

**Diagram:**
```text
┌─────────┐   ┌───────────┐   ┌──────────────┐   ┌──────────────┐
│ Prompt  │→→│ LLM Answer │→→│ Reflection    │→→│ Final Output  │
└─────────┘   └────┬──────┘   └────┬─────────┘   └────┬─────────┘
                   │ critique       │ revise          │ return
                   ↓                ↓                 ↓
               Tool Calls ←─────────┘ (if needed)
```

**Code (ReAct-style prompt sketch):**
```python
react_prompt = ChatPromptTemplate.from_messages([
    ("system", "Follow ReAct: Thought -> Action -> Observation."),
    ("user", "{input}")
])
react_chain = react_prompt | llm | StrOutputParser()
```

**Tips:**
- Add a reflection pass: LLM critiques its own answer before finalizing.
- In graphs, model tool calls as nodes; capture observations in state.
- Limit action space; enforce JSON for actions.

## Best Practices
**Brief:** Guardrails, testing, logging, error handling.

**Diagram (Streaming vs Batch):**
```text
Streaming:                Batch:
┌────┐ tok tok tok → ┌──────────┐   ┌────┐ full response → ┌──────────┐
│LLM │──────────────→│ Client   │   │LLM │────────────────→│ Client   │
└────┘                └──────────┘   └────┘                 └──────────┘
```

**Tips:**
- Validate inputs/outputs; sanitize tool results.
- Add retries with backoff for transient API errors.
- Unit-test prompts/parsers; integration-test full chains.
- Log prompts + responses securely (redact PII).
- Monitor latency, cost, and error rates continuously.
```
