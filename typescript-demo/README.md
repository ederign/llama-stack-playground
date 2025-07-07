# LlamaStack RAG Agent Demo - TypeScript Version

This is a TypeScript conversion of the Jupyter notebook `../myagent.ipynb` that demonstrates how to create a RAG (Retrieval-Augmented Generation) agent using LlamaStack.

## Features

- 🗄️ Creates a FAISS vector database
- 📄 Ingests dog breed documents
- 🛠️ Registers RAG toolgroup
- 🤖 Creates an agent with RAG capabilities
- 🧪 Tests the agent with dog breed questions
- 📋 Shows tool usage and responses

## Prerequisites

1. **LlamaStack Server**: Make sure you have LlamaStack running on `http://localhost:8321`
2. **Node.js**: Version 18 or higher
3. **npm**: For package management

## Setup

1. **Navigate to the typescript-demo directory**:

   ```bash
   cd typescript-demo
   ```

2. **Install dependencies**:

   ```bash
   npm install
   ```

3. **Install TypeScript dependencies** (if not already installed):
   ```bash
   npm install -g typescript tsx
   ```

## Running the Demo

### Option 1: Direct execution with tsx

```bash
npm run dev
```

### Option 2: Using the shebang

```bash
./agents.ts
```

### Option 3: Build and run

```bash
npm run build
npm start
```

## What the Demo Does

1. **Lists available models** and selects an appropriate one (prefers Ollama if available)
2. **Creates a vector database** using FAISS with the `all-MiniLM-L6-v2` embedding model
3. **Inserts documents** about dog breeds:
   - Bella is a Cavalier King Charles Spaniel
   - Dora is a Pug
4. **Registers a RAG toolgroup** that connects to the vector database
5. **Creates an agent** configured to use the RAG toolgroup
6. **Tests the agent** with questions about dog breeds
7. **Shows tool usage** and responses

## Expected Output

```
🚀 Starting LlamaStack RAG Agent Demo...

📋 Listing available models...
Found X models
✅ Using model: ollama/llama3.2:3b

🗄️ Creating vector database...
✅ Created vector DB: toy_faiss_db

📄 Inserting documents into vector database...
✅ Documents ingested.

🔧 Listing providers...
✅ rag-runtime provider found.

🛠️ Registering RAG toolgroup...
✅ RAG toolgroup registered.

🤖 Creating agent with RAG capabilities...
✅ Agent created with ID: [agent-id]

🧪 Testing agent with dog breed questions...

❓ Question: Which breed is Bella?
🤖 Answer: Bella is a Cavalier King Charles Spaniel.

❓ Question: Which breed is Dora?
🤖 Answer: Dora is a Pug.

📋 Listing registered toolgroups...
🧠 RAG Toolgroup: rag-dogs
   Args: {"vectorDbIds":["toy_faiss_db"]}

🎉 Demo completed successfully!
```

## Configuration

The demo uses the following configuration:

- **Base URL**: `http://localhost:8321`
- **Vector DB**: FAISS with `all-MiniLM-L6-v2` embeddings
- **Model**: Prefers `ollama/llama3.2:3b`, falls back to first available LLM
- **Agent Instructions**: "Always use retrieval tool to fetch info about dog breeds before answering."

## Troubleshooting

1. **Connection Error**: Make sure LlamaStack is running on `http://localhost:8321`
2. **Model Not Found**: The demo will automatically select an available model
3. **RAG Provider Missing**: Ensure the `rag-runtime` provider is available in your LlamaStack setup

## Files

- `agents.ts` - Main TypeScript implementation
- `package.json` - Node.js dependencies and scripts
- `tsconfig.json` - TypeScript configuration
- `README.md` - This documentation

## Comparison with Python Version

This TypeScript version replicates the functionality of `../myagent.ipynb`:

- ✅ Vector database creation
- ✅ Document ingestion
- ✅ RAG toolgroup registration
- ✅ Agent creation and testing
- ✅ Tool usage logging

The main differences are:

- Uses TypeScript/Node.js instead of Python
- More structured error handling
- Better logging with emojis
- Modular function structure

## Project Structure

```
playground/
├── typescript-demo/          # This TypeScript demo
│   ├── agents.ts            # Main implementation
│   ├── package.json         # Node.js dependencies
│   ├── tsconfig.json        # TypeScript config
│   └── README.md           # This file
├── myagent.ipynb           # Original Python notebook
├── pyproject.toml          # Python dependencies
└── ...                     # Other Python files
```
