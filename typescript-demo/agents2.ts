#!/usr/bin/env -S npm run tsn -T

// @ts-ignore - Import issue with llama-stack-client types
import { LlamaStackClient } from "llama-stack-client";

// Client setup
const client = new LlamaStackClient({
  baseURL: "http://localhost:8321",
});

// Direct HTTP client for calling server without using the client library
async function listModelsDirectly() {
  try {
    console.log("🌐 Calling LlamaStack server directly via HTTP...\n");

    const baseURL = "http://localhost:8321";
    const response = await fetch(`${baseURL}/v1/models`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const models = await response.json();
    console.log(`📋 Found ${models.length} models via direct HTTP call:\n`);

    // Filter for LLM models (excluding guards and large models)
    const availableModels = models.filter(
      (model: any) =>
        model.model_type === "llm" &&
        !model.identifier.includes("guard") &&
        !model.identifier.includes("405")
    );

    console.log(`Available LLM models: ${availableModels.length}`);
    for (const model of availableModels) {
      console.log(`  - ${model.identifier} (${model.model_type})`);
      console.log(`    Provider: ${model.provider_id}`);
      console.log(`    Type: ${model.model_type}`);
      console.log("");
    }

    return models;
  } catch (error) {
    console.error("❌ Error calling server directly:", error);
    throw error;
  }
}

// Direct HTTP client for listing toolgroups
async function listToolgroupsDirectly() {
  try {
    console.log("🛠️ Listing toolgroups via direct HTTP call...\n");

    const baseURL = "http://localhost:8321";
    const response = await fetch(`${baseURL}/v1/toolgroups`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const toolgroups = await response.json();
    console.log(
      `Found ${toolgroups.length} toolgroups via direct HTTP call:\n`
    );

    for (const tg of toolgroups) {
      console.log(`🔧 Toolgroup: ${tg.identifier}`);
      console.log(`   Provider: ${tg.provider_id}`);
      if (tg.args) {
        console.log(`   Args: ${JSON.stringify(tg.args)}`);
      }
      console.log("");
    }

    return toolgroups;
  } catch (error) {
    console.error("❌ Error listing toolgroups directly:", error);
    throw error;
  }
}

// Direct HTTP client for listing vector databases
async function listVectorDBsDirectly() {
  try {
    console.log("🗄️ Listing vector databases via direct HTTP call...\n");

    const baseURL = "http://localhost:8321";
    const response = await fetch(`${baseURL}/v1/vector_dbs`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const vectorDbs = await response.json();
    console.log(
      `Found ${vectorDbs.length} vector databases via direct HTTP call:\n`
    );

    for (const vdb of vectorDbs) {
      console.log(`📚 Vector DB: ${vdb.identifier}`);
      console.log(`   Provider: ${vdb.provider_id}`);
      console.log(`   Embedding Model: ${vdb.embedding_model}`);
      console.log(`   Dimension: ${vdb.embedding_dimension}`);
      console.log("");
    }

    return vectorDbs;
  } catch (error) {
    console.error("❌ Error listing vector databases directly:", error);
    throw error;
  }
}

// Direct HTTP client for listing providers
async function listProvidersDirectly() {
  try {
    console.log("🔧 Listing providers via direct HTTP call...\n");

    const baseURL = "http://localhost:8321";
    const response = await fetch(`${baseURL}/v1/providers`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const providers = await response.json();
    console.log(`Found ${providers.length} providers via direct HTTP call:\n`);

    for (const provider of providers) {
      console.log(`⚙️  Provider: ${provider.provider_id}`);
      console.log(`   Type: ${provider.provider_type}`);
      console.log("");
    }

    return providers;
  } catch (error) {
    console.error("❌ Error listing providers directly:", error);
    throw error;
  }
}

async function main() {
  try {
    console.log("🚀 Starting LlamaStack Agent Lister...\n");

    // Step 1: List available models using direct HTTP calls
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

    console.log(`Available LLM models: ${availableModels.length}`);
    for (const model of availableModels) {
      console.log(`  - ${model.identifier} (${model.model_type})`);
    }
    console.log("");

    // Step 2: List existing agents
    console.log("🤖 Listing existing agents...");
    // Note: The agents.list() method might not exist in the current API
    // We'll try to get agent information through other means
    console.log(
      "⚠️  Note: Direct agent listing may not be available in current API"
    );
    console.log(
      "   You can check for agents by looking at sessions and toolgroups\n"
    );

    // Step 3: List toolgroups to see what's configured
    console.log("🛠️ Listing registered toolgroups...");
    const toolgroups = await client.toolgroups.list();
    console.log(`Found ${toolgroups.length} toolgroups:\n`);

    for (const tg of toolgroups) {
      console.log(`🔧 Toolgroup: ${tg.identifier}`);
      console.log(`   Provider: ${tg.provider_id}`);
      if (tg.args) {
        console.log(`   Args: ${JSON.stringify(tg.args)}`);
      }
      console.log("");
    }

    // Step 4: List vector databases
    console.log("🗄️ Listing vector databases...");
    const vectorDbs = await client.vectorDBs.list();
    console.log(`Found ${vectorDbs.length} vector databases:\n`);

    for (const vdb of vectorDbs) {
      console.log(`📚 Vector DB: ${vdb.identifier}`);
      console.log(`   Provider: ${vdb.provider_id}`);
      console.log(`   Embedding Model: ${vdb.embedding_model}`);
      console.log(`   Dimension: ${vdb.embedding_dimension}`);
      console.log("");
    }

    // Step 5: List providers
    console.log("🔧 Listing providers...");
    const providers = await client.providers.list();
    console.log(`Found ${providers.length} providers:\n`);

    for (const provider of providers) {
      console.log(`⚙️  Provider: ${provider.provider_id}`);
      console.log(`   Type: ${provider.provider_type}`);
      console.log("");
    }

    console.log("🎉 Agent listing completed successfully!");
  } catch (error) {
    console.error("❌ Error occurred:", error);
    if (error instanceof Error) {
      console.error("Error details:", error.message);
    }
  }
}

// New main function that uses direct HTTP calls
async function mainDirect() {
  try {
    console.log("🚀 Starting LlamaStack Direct HTTP Lister...\n");

    // Use direct HTTP calls instead of client library
    await listModelsDirectly();
    await listToolgroupsDirectly();
    await listVectorDBsDirectly();
    await listProvidersDirectly();

    console.log("🎉 Direct HTTP listing completed successfully!");
  } catch (error) {
    console.error("❌ Error occurred:", error);
    if (error instanceof Error) {
      console.error("Error details:", error.message);
    }
  }
}

// Run the main function
// main().catch(console.error);

// Uncomment the line below to use direct HTTP calls instead
mainDirect().catch(console.error);
