# Agentic RAG System Architecture

## System Overview
This system implements a **Multi-Agent Retrieval-Augmented Generation (RAG) chatbot** that processes various document formats and answers user queries through coordinated agent interactions using the Model Context Protocol (MCP).

## Core Architecture Components

### 1. Model Context Protocol (MCP) Implementation
```python
class MCPMessage:
    def __init__(self, sender: str, receiver: str, msg_type: str, 
                 trace_id: str = None, payload: Dict = None):
        self.sender = sender          # Agent sending the message
        self.receiver = receiver      # Target agent
        self.type = msg_type         # Message type (e.g., "INGESTION_REQUEST")
        self.trace_id = trace_id     # Unique trace for request tracking
        self.payload = payload       # Message data/context
        self.timestamp = datetime.now().isoformat()
```

**Key MCP Features:**
- **Structured Communication**: All inter-agent communication follows standardized message format
- **Trace Tracking**: Each request has a unique trace_id for end-to-end tracking
- **Type-based Routing**: Message types determine how agents process requests
- **Payload Context**: Rich context data passed between agents

### 2. Message Bus (Communication Layer)
```python
class MessageBus:
    def __init__(self):
        self.messages = []           # Message history
        self.subscribers = {}        # Agent subscriptions
    
    def publish(self, message: MCPMessage):      # Send message
    def subscribe(self, agent_name: str, callback):  # Register agent
```

**Features:**
- **Pub/Sub Pattern**: Agents subscribe to receive relevant messages
- **In-Memory Communication**: Fast, synchronous message passing
- **Message History**: All communications are logged for debugging

## Agent Architecture

### 1. IngestionAgent
**Responsibility**: Document parsing and preprocessing
```python
class IngestionAgent:
    # Supported formats: PDF, PPTX, CSV, DOCX, TXT/MD
    def parse_pdf(self, file_path: str) -> str
    def parse_pptx(self, file_path: str) -> str
    def parse_csv(self, file_path: str) -> str
    def parse_docx(self, file_path: str) -> str
    def parse_txt(self, file_path: str) -> str
```

**MCP Messages Handled:**
- **Receives**: `INGESTION_REQUEST` (with file paths)
- **Sends**: `INGESTION_COMPLETE` (with processed documents)

**Process Flow:**
1. Receives files from coordinator
2. Determines file type and applies appropriate parser
3. Splits text into chunks using RecursiveCharacterTextSplitter
4. Creates Document objects with metadata
5. Sends processed documents to RetrievalAgent

### 2. RetrievalAgent
**Responsibility**: Embedding generation and semantic retrieval
```python
class RetrievalAgent:
    def __init__(self):
        self.vector_store = None  # FAISS vector store
        # Uses HuggingFace embeddings: "sentence-transformers/all-MiniLM-L6-v2"
    
    def create_vector_store(self, documents)    # Build FAISS index
    def retrieve_context(self, query, k=3)      # Semantic search
```

**MCP Messages Handled:**
- **Receives**: `INGESTION_COMPLETE` (creates vector store)
- **Receives**: `RETRIEVAL_REQUEST` (performs search)
- **Sends**: `VECTORSTORE_READY`, `CONTEXT_RESPONSE`

**Process Flow:**
1. Creates FAISS vector store from ingested documents
2. Performs similarity search for user queries
3. Returns top-k relevant chunks with source metadata

### 3. LLMResponseAgent
**Responsibility**: LLM interaction and response generation
```python
class LLMResponseAgent:
    def __init__(self):
        # Uses HuggingFace Inference API with Llama-3.1-8B-Instruct
        self.client = InferenceClient(model="meta-llama/Llama-3.1-8B-Instruct")
    
    def generate_response(self, query, context)  # Stream LLM response
```

**MCP Messages Handled:**
- **Receives**: `CONTEXT_RESPONSE` (with retrieved chunks)
- **Sends**: `LLM_RESPONSE_STREAM` (streaming response)

**Process Flow:**
1. Receives query and retrieved context
2. Builds conversational prompt with context
3. Calls LLM API with streaming enabled
4. Returns streaming response tokens

### 4. CoordinatorAgent
**Responsibility**: Workflow orchestration and user interface
```python
class CoordinatorAgent:
    def process_files(self, files)              # Initiate document processing
    def handle_query(self, query) -> Generator  # Process user queries
```

**MCP Messages Handled:**
- **Sends**: `INGESTION_REQUEST`, `RETRIEVAL_REQUEST`
- **Receives**: `VECTORSTORE_READY`, `LLM_RESPONSE_STREAM`

**Process Flow:**
1. Manages overall system state
2. Coordinates file processing workflow
3. Handles user queries end-to-end
4. Manages streaming responses to UI

## System Workflow

### Document Processing Flow
```
User Upload → CoordinatorAgent → [MCP: INGESTION_REQUEST] → IngestionAgent
                                                              ↓
                                                        Parse & Chunk Documents
                                                              ↓
RetrievalAgent ← [MCP: INGESTION_COMPLETE] ← Documents with Metadata
     ↓
Create FAISS Vector Store
     ↓
CoordinatorAgent ← [MCP: VECTORSTORE_READY] ← Status Update
```

### Query Processing Flow
```
User Query → CoordinatorAgent → [MCP: RETRIEVAL_REQUEST] → RetrievalAgent
                                                              ↓
                                                        Semantic Search
                                                              ↓
LLMResponseAgent ← [MCP: CONTEXT_RESPONSE] ← Retrieved Chunks + Metadata
     ↓
Generate LLM Prompt with Context
     ↓
Stream Response from Llama-3.1-8B
     ↓
CoordinatorAgent ← [MCP: LLM_RESPONSE_STREAM] ← Streaming Tokens
     ↓
User Interface (Gradio)
```

## Technical Stack

### Core Technologies
- **Framework**: Python with Gradio for UI
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: HuggingFace sentence-transformers/all-MiniLM-L6-v2
- **LLM**: Meta Llama-3.1-8B-Instruct via HuggingFace Inference API
- **Document Processing**: PyPDF2, python-pptx, pandas, python-docx

### Document Format Support
| Format | Library | Use Case |
|--------|---------|----------|
| PDF | PyPDF2 | Research papers, reports |
| PPTX | python-pptx | Presentations, slides |
| CSV | pandas | Data tables, metrics |
| DOCX | python-docx | Word documents |
| TXT/MD | Built-in | Plain text, markdown |

### MCP Message Types
```python
# Message types used in the system
INGESTION_REQUEST    # Start document processing
INGESTION_COMPLETE   # Documents processed and chunked
RETRIEVAL_REQUEST    # Perform semantic search
CONTEXT_RESPONSE     # Retrieved chunks with context
VECTORSTORE_READY    # Vector database is ready
LLM_RESPONSE_STREAM  # Streaming LLM response
```

## Key Features

### 1. Streaming Response
- **Real-time Updates**: Users see response tokens as they're generated
- **Better UX**: No waiting for complete response
- **Resource Efficient**: Immediate feedback without blocking

### 2. Multi-Format Support
- **Unified Processing**: Different parsers for each format
- **Metadata Preservation**: Source tracking for citations
- **Error Handling**: Graceful failure for unsupported files

### 3. Contextual RAG
- **Semantic Search**: Vector similarity for relevant chunks
- **Source Attribution**: Documents are cited in responses
- **Context Window**: Top-k chunks provide comprehensive context

### 4. Agent Autonomy
- **Separation of Concerns**: Each agent has specific responsibilities
- **Scalable Architecture**: Easy to add new agents or modify existing ones
- **Message-Driven**: Loose coupling through MCP communication

## Architecture Benefits

### 1. Modularity
- **Independent Agents**: Each agent can be developed, tested, and deployed separately
- **Easy Maintenance**: Changes to one agent don't affect others
- **Technology Flexibility**: Different agents can use different technologies

### 2. Scalability
- **Horizontal Scaling**: Agents can be distributed across multiple processes/servers
- **Load Distribution**: Different agents can handle different aspects of the workload
- **Resource Optimization**: Each agent can be optimized for its specific task

### 3. Observability
- **Message Tracing**: Every request can be tracked end-to-end
- **Debugging**: Clear message flow makes debugging easier
- **Monitoring**: Agent performance can be monitored independently

### 4. Extensibility
- **New Document Types**: Easy to add new parsers to IngestionAgent
- **Multiple LLMs**: LLMResponseAgent can support different models
- **New Retrieval Methods**: RetrievalAgent can incorporate different search strategies

## Performance Characteristics

### Document Processing
- **Chunk Size**: 1000 characters with 200 character overlap
- **Embedding Dimension**: 384 (all-MiniLM-L6-v2)
- **Search Results**: Top-3 most relevant chunks by default

### Response Generation
- **Model**: Llama-3.1-8B-Instruct (8 billion parameters)
- **Max Tokens**: 512 per response
- **Temperature**: 0.7 for balanced creativity/accuracy
- **Streaming**: Token-by-token response delivery

This architecture successfully implements the MCP-based agentic RAG system with clear separation of concerns, structured communication, and comprehensive document support.
