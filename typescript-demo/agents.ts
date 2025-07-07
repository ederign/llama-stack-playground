#!/usr/bin/env -S npm run tsn -T

// @ts-ignore - Import issue with llama-stack-client types
import { LlamaStackClient } from "llama-stack-client";

// Client setup
const client = new LlamaStackClient({
  baseURL: "http://localhost:8321",
});

async function main() {
  try {
    console.log("🚀 Starting LlamaStack RAG Agent Demo...\n");

    // Step 1: List available models and select one
    console.log("📋 Listing available models...");
    const models = await client.models.list();
    console.log(`Found ${models.length} models`);

    // Filter for LLM models (excluding guards and large models)
    const availableModels = models.filter(
      (model: any) =>
        model.model_type === "llm" &&
        !model.identifier.includes("guard") &&
        !model.identifier.includes("405")
    );

    if (availableModels.length === 0) {
      console.log("❌ No available models. Exiting.");
      return;
    }

    // Prefer Ollama model if available, otherwise use first available
    const selectedModel =
      availableModels.find((m: any) => m.identifier === "ollama/llama3.2:3b")
        ?.identifier || availableModels[0].identifier;
    console.log(`✅ Using model: ${selectedModel}\n`);

    const vectorDbId = "toy_faiss_db_node";
    // Step 2: Create vector database
    console.log("🗄️ Creating vector database...");
    const vectorDb = await client.vectorDBs.register({
      vector_db_id: vectorDbId,
      provider_id: "faiss",
      embedding_model: "all-MiniLM-L6-v2",
    });
    console.log(`✅ Created vector DB: ${vectorDb.identifier}\n`);

    // List available knowledge sources
    await listKnowledgeSources();

    // List available MCP servers
    await listMCPServers();

    // Step 3: Insert documents into vector database
    console.log("📄 Inserting documents into vector database...");
    const docs = [
      {
        document_id: "dog1",
        content: "Bella breed is a cavalier king.",
        mime_type: "text/plain",
        metadata: {},
      },
      {
        document_id: "dog2",
        content: "Dora breed is a pug.",
        mime_type: "text/plain",
        metadata: {},
      },
      {
        document_id: "dog3",
        content: "Bento breed is a pug.",
        mime_type: "text/plain",
        metadata: {},
      },
    ];

    await client.toolRuntime.ragTool.insert({
      documents: docs,
      vector_db_id: vectorDbId,
      chunk_size_in_tokens: 128,
    });
    console.log("✅ Documents ingested.\n");

    // Step 4: List providers to verify rag-runtime is available
    console.log("🔧 Listing providers...");
    const providers = await client.providers.list();
    const ragProvider = providers.find(
      (p: any) => p.provider_id === "rag-runtime"
    );
    if (!ragProvider) {
      console.log("❌ rag-runtime provider not found. Exiting.");
      return;
    }
    console.log("✅ rag-runtime provider found.\n");

    // Step 5: Register RAG toolgroup
    console.log("🛠️ Registering RAG toolgroup...");
    await client.toolgroups.register({
      toolgroup_id: "rag-dogs_node",
      provider_id: "rag-runtime",
      args: { vector_db_ids: [vectorDbId] },
    });
    console.log("✅ RAG toolgroup registered.\n");

    // Step 6: Create agent with RAG capabilities
    console.log("🤖 Creating agent with RAG capabilities...");
    const agentConfig: any = {
      model: selectedModel,
      instructions:
        "Always use retrieval tool to fetch info about dog breeds before answering.",
      sampling_params: {
        strategy: { type: "top_p", temperature: 0.7, top_p: 0.9 },
      },
      toolgroups: [
        {
          name: "builtin::rag/knowledge_search",
          args: { vector_db_ids: [vectorDbId] },
        },
      ],
      tool_choice: "auto",
      tool_prompt_format: "python_list",
      input_shields: [],
      output_shields: [],
      enable_session_persistence: false,
      max_infer_iters: 10,
    };

    const agentCreateResponse = await client.agents.create({
      agent_config: agentConfig,
    });
    const agentId = agentCreateResponse.agent_id;
    console.log(`✅ Agent created with ID: ${agentId}\n`);

    // Step 7: Test the agent with dog breed questions
    console.log("🧪 Testing agent with dog breed questions...\n");

    const testQuestions = [
      "Which breed is Bella?",
      "Which breed is Dora?",
      "Which breed is Bento?",
    ];

    for (const question of testQuestions) {
      console.log(`❓ Question: ${question}`);

      // Create session
      const sessionResponse = await client.agents.session.create(agentId, {
        session_name: "dog-chat",
      });
      const sessionId = sessionResponse.session_id;

      // Create turn with RAG toolgroup
      const turnResponse = await client.agents.turn.create(agentId, sessionId, {
        stream: true,
        messages: [
          {
            role: "user",
            content: question,
          },
        ],
      });

      // Handle streaming response
      let fullResponse = "";
      for await (const chunk of turnResponse) {
        //  console.log("STREAM CHUNK:", JSON.stringify(chunk, null, 2));
        if (!chunk || !chunk.event) continue;
        if (
          chunk.event.payload &&
          chunk.event.payload.event_type === "turn_complete"
        ) {
          const outputMessage = chunk.event.payload.turn?.output_message;
          if (
            outputMessage?.content &&
            typeof outputMessage.content === "string"
          ) {
            fullResponse = outputMessage.content;
          } else if (Array.isArray(outputMessage?.content)) {
            // Handle array of content items
            fullResponse = outputMessage.content
              .filter((item: any) => item.type === "text")
              .map((item: any) => item.text)
              .join("");
          } else {
            fullResponse = "No response content";
          }
          break;
        }
      }

      console.log(`🤖 Answer: ${fullResponse}\n`);
    }

    // Step 8: Test Bento specifically
    console.log("🐕 Testing Bento question specifically...\n");
    await askAboutBento(agentId);

    // Step 9: Test Leao specifically
    console.log("🦁 Testing Leao question specifically...\n");
    await askAboutLeao(agentId);

    // Step 10: List toolgroups to verify setup
    console.log("📋 Listing registered toolgroups...");
    const toolgroups = await client.toolgroups.list();
    for (const tg of toolgroups) {
      if (tg.provider_id === "rag-runtime") {
        console.log(`🧠 RAG Toolgroup: ${tg.identifier}`);
        console.log(`   Args: ${JSON.stringify(tg.args)}\n`);
      }
    }

    console.log("🎉 Demo completed successfully!");
  } catch (error) {
    console.error("❌ Error occurred:", error);
    if (error instanceof Error) {
      console.error("Error details:", error.message);
    }
  }
}

// Dedicated method to ask about Bento
async function askAboutBento(agentId: string) {
  try {
    console.log("❓ Question: What breed is Bento?");

    // Create session
    const sessionResponse = await client.agents.session.create(agentId, {
      session_name: "bento-chat",
    });
    const sessionId = sessionResponse.session_id;

    // Create turn with RAG toolgroup
    const turnResponse = await client.agents.turn.create(agentId, sessionId, {
      stream: true,
      messages: [
        {
          role: "user",
          content: "What breed is Bento?",
        },
      ],
    });

    // Handle streaming response
    let fullResponse = "";
    for await (const chunk of turnResponse) {
      if (!chunk || !chunk.event) continue;
      if (
        chunk.event.payload &&
        chunk.event.payload.event_type === "turn_complete"
      ) {
        const outputMessage = chunk.event.payload.turn?.output_message;
        if (
          outputMessage?.content &&
          typeof outputMessage.content === "string"
        ) {
          fullResponse = outputMessage.content;
        } else if (Array.isArray(outputMessage?.content)) {
          // Handle array of content items
          fullResponse = outputMessage.content
            .filter((item: any) => item.type === "text")
            .map((item: any) => item.text)
            .join("");
        } else {
          fullResponse = "No response content";
        }
        break;
      }
    }

    console.log(`🤖 Answer: ${fullResponse}\n`);
  } catch (error) {
    console.error("❌ Error asking about Bento:", error);
  }
}

// Dedicated method to ask about Leao
async function askAboutLeao(agentId: string) {
  try {
    console.log("❓ Question: What breed is Joao?");
    // Create session
    const sessionResponse = await client.agents.session.create(agentId, {
      session_name: "leao-chat",
    });
    const sessionId = sessionResponse.session_id;

    // Create turn with RAG toolgroup
    const turnResponse = await client.agents.turn.create(agentId, sessionId, {
      stream: true,
      messages: [
        {
          role: "user",
          content: "What breed is Joao?",
        },
      ],
    });

    // Handle streaming response
    let fullResponse = "";
    for await (const chunk of turnResponse) {
      if (!chunk || !chunk.event) continue;
      if (
        chunk.event.payload &&
        chunk.event.payload.event_type === "turn_complete"
      ) {
        const outputMessage = chunk.event.payload.turn?.output_message;
        if (
          outputMessage?.content &&
          typeof outputMessage.content === "string"
        ) {
          fullResponse = outputMessage.content;
        } else if (Array.isArray(outputMessage?.content)) {
          // Handle array of content items
          fullResponse = outputMessage.content
            .filter((item: any) => item.type === "text")
            .map((item: any) => item.text)
            .join("");
        } else {
          fullResponse = "No response content";
        }
        break;
      }
    }

    console.log(`🤖 Answer: ${fullResponse}\n`);
  } catch (error) {
    console.error("❌ Error asking about Leao:", error);
  }
}

// Method to list available knowledge sources (vector databases)
async function listKnowledgeSources() {
  try {
    console.log("📚 Listing available knowledge sources...");
    const vectorDbs = await client.vectorDBs.list();
    console.log(`Found ${vectorDbs.length} knowledge sources:`);

    for (const vdb of vectorDbs) {
      console.log(`  - ${vdb.identifier} (${vdb.provider_id})`);
      console.log(`    Embedding model: ${vdb.embedding_model}`);
      console.log(`    Dimension: ${vdb.embedding_dimension}`);
      console.log("");
    }
  } catch (error) {
    console.error("❌ Error listing knowledge sources:", error);
  }
}

// Method to list available MCP servers
async function listMCPServers() {
  try {
    console.log("🔌 Listing available MCP servers...");

    // Check toolgroups for MCP endpoints
    const toolgroups = await client.toolgroups.list();
    const mcpToolgroups = toolgroups.filter(
      (tg: any) =>
        tg.provider_id === "model-context-protocol" || tg.mcp_endpoint
    );

    console.log(`Found ${mcpToolgroups.length} MCP toolgroups:`);

    for (const tg of mcpToolgroups) {
      console.log(`  - ${tg.identifier} (${tg.provider_id})`);
      if (tg.mcp_endpoint) {
        console.log(`    MCP Endpoint: ${tg.mcp_endpoint.uri}`);
      }
      console.log("");
    }

    if (mcpToolgroups.length === 0) {
      console.log("  No MCP servers found.");
    }
  } catch (error) {
    console.error("❌ Error listing MCP servers:", error);
  }
}

// Run the main function
main().catch(console.error);
