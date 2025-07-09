package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// APIResponse represents a generic API response
type APIResponse struct {
	AgentID string `json:"agent_id,omitempty"`
	ID      string `json:"id,omitempty"`
	Object  string `json:"object,omitempty"`
	Created int64  `json:"created,omitempty"`
	Model   string `json:"model,omitempty"`
	Choices []struct {
		Index        int    `json:"index"`
		FinishReason string `json:"finish_reason"`
		Message      struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices,omitempty"`
}

// FileResponse represents a file upload response
type FileResponse struct {
	ID        string `json:"id"`
	Object    string `json:"object"`
	Bytes     int    `json:"bytes"`
	CreatedAt int64  `json:"created_at"`
	Filename  string `json:"filename"`
	Purpose   string `json:"purpose"`
}

// VectorStore represents a vector store
type VectorStore struct {
	ID         string                 `json:"id"`
	Object     string                 `json:"object"`
	Name       string                 `json:"name"`
	CreatedAt  int64                  `json:"created_at"`
	FileCounts map[string]int         `json:"file_counts"`
	Metadata   map[string]interface{} `json:"metadata"`
	Status     string                 `json:"status"`
	ExpiresAt  *int64                 `json:"expires_at,omitempty"`
	LastUsedAt *int64                 `json:"last_used_at,omitempty"`
}

// VectorStoreFile represents a file attached to a vector store
type VectorStoreFile struct {
	ID               string                 `json:"id"`
	Object           string                 `json:"object"`
	CreatedAt        int64                  `json:"created_at"`
	VectorStoreID    string                 `json:"vector_store_id"`
	Status           string                 `json:"status"`
	UsageBytes       int                    `json:"usage_bytes"`
	Attributes       map[string]interface{} `json:"attributes"`
	ChunkingStrategy interface{}            `json:"chunking_strategy"`
	LastError        *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"last_error,omitempty"`
}

// Document represents a document for RAG operations
type Document struct {
	Content    interface{}            `json:"content"`
	DocumentID string                 `json:"document_id"`
	Metadata   map[string]interface{} `json:"metadata"`
	MimeType   string                 `json:"mime_type,omitempty"`
}

// RagToolInsertParams represents parameters for RAG tool insert
type RagToolInsertParams struct {
	ChunkSizeInTokens int        `json:"chunk_size_in_tokens"`
	Documents         []Document `json:"documents"`
	VectorDBID        string     `json:"vector_db_id"`
}

// AgentConfig represents the configuration for creating an agent
type AgentConfig struct {
	Instructions string                   `json:"instructions"`
	Model        string                   `json:"model"`
	Name         string                   `json:"name,omitempty"`
	Description  string                   `json:"description,omitempty"`
	Tools        []map[string]interface{} `json:"tools,omitempty"`
	Memory       map[string]interface{}   `json:"memory,omitempty"`

	// Additional fields from TypeScript AgentConfig
	SamplingParams           *SamplingParams `json:"sampling_params,omitempty"`
	ToolChoice               string          `json:"tool_choice,omitempty"`
	ToolPromptFormat         string          `json:"tool_prompt_format,omitempty"`
	InputShields             []string        `json:"input_shields,omitempty"`
	OutputShields            []string        `json:"output_shields,omitempty"`
	EnableSessionPersistence bool            `json:"enable_session_persistence,omitempty"`
	MaxInferIters            int             `json:"max_infer_iters,omitempty"`
	Toolgroups               []interface{}   `json:"toolgroups,omitempty"`
}

// SamplingParams represents the sampling parameters for the agent
type SamplingParams struct {
	Strategy          SamplingStrategy `json:"strategy"`
	MaxTokens         *int             `json:"max_tokens,omitempty"`
	RepetitionPenalty *float64         `json:"repetition_penalty,omitempty"`
	Stop              []string         `json:"stop,omitempty"`
}

// SamplingStrategy represents the sampling strategy
type SamplingStrategy struct {
	Type        string   `json:"type"`
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	TopK        *int     `json:"top_k,omitempty"`
}

// AgentCreateParams represents the parameters for creating an agent
type AgentCreateParams struct {
	AgentConfig AgentConfig `json:"agent_config"`
}

// Message represents a chat message
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	Name    string `json:"name,omitempty"`
}

// ChatCompletionParams represents the parameters for creating a chat completion
type ChatCompletionParams struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Temperature *float64  `json:"temperature,omitempty"`
	MaxTokens   *int      `json:"max_tokens,omitempty"`
	Stream      *bool     `json:"stream,omitempty"`
}

// LlamaStackClient represents a client for the Llama Stack API
type LlamaStackClient struct {
	BaseURL    string
	HTTPClient *http.Client
	APIKey     string
}

// NewLlamaStackClient creates a new Llama Stack client
func NewLlamaStackClient(baseURL, apiKey string) *LlamaStackClient {
	return &LlamaStackClient{
		BaseURL: baseURL,
		HTTPClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		APIKey: apiKey,
	}
}

// UploadFile uploads a file to the Llama Stack API
func (c *LlamaStackClient) UploadFile(ctx context.Context, filePath, purpose string) (*FileResponse, error) {
	// Open the file
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	// Create a buffer to store the multipart form data
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	// Create the file field
	part, err := writer.CreateFormFile("file", filepath.Base(filePath))
	if err != nil {
		return nil, fmt.Errorf("failed to create form file: %w", err)
	}

	// Copy file content to the form field
	_, err = io.Copy(part, file)
	if err != nil {
		return nil, fmt.Errorf("failed to copy file content: %w", err)
	}

	// Add the purpose field
	err = writer.WriteField("purpose", purpose)
	if err != nil {
		return nil, fmt.Errorf("failed to write purpose field: %w", err)
	}

	// Close the writer
	err = writer.Close()
	if err != nil {
		return nil, fmt.Errorf("failed to close writer: %w", err)
	}

	// Create the request
	url := c.BaseURL + "/v1/openai/v1/files"
	req, err := http.NewRequestWithContext(ctx, "POST", url, &buf)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", writer.FormDataContentType())
	req.Header.Set("Authorization", "Bearer "+c.APIKey)

	fmt.Println("=== REST CALL: Upload File ===")
	fmt.Printf("URL: %s\n", url)
	fmt.Printf("Method: %s\n", req.Method)
	fmt.Printf("Headers: %v\n", req.Header)
	fmt.Printf("File: %s\n", filePath)
	fmt.Printf("Purpose: %s\n", purpose)

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	fmt.Printf("Response Status: %s\n", resp.Status)
	fmt.Printf("Response Headers: %v\n", resp.Header)

	body, _ := io.ReadAll(resp.Body)
	fmt.Printf("Response Body:\n%s\n", string(body))
	fmt.Println("=== END REST CALL ===\n")

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var response FileResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &response, nil
}

// CreateVectorStore creates a new vector store
func (c *LlamaStackClient) CreateVectorStore(ctx context.Context, name string, metadata map[string]interface{}) (*VectorStore, error) {
	payload := map[string]interface{}{
		"name":     name,
		"metadata": metadata,
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal vector store params: %w", err)
	}

	url := c.BaseURL + "/v1/openai/v1/vector_stores"
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.APIKey)

	fmt.Println("=== REST CALL: Create Vector Store ===")
	fmt.Printf("URL: %s\n", url)
	fmt.Printf("Method: %s\n", req.Method)
	fmt.Printf("Headers: %v\n", req.Header)
	fmt.Printf("Request Body:\n%s\n", string(jsonData))

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	fmt.Printf("Response Status: %s\n", resp.Status)
	fmt.Printf("Response Headers: %v\n", resp.Header)

	body, _ := io.ReadAll(resp.Body)
	fmt.Printf("Response Body:\n%s\n", string(body))
	fmt.Println("=== END REST CALL ===\n")

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var response VectorStore
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &response, nil
}

// AttachFileToVectorStore attaches a file to a vector store
func (c *LlamaStackClient) AttachFileToVectorStore(ctx context.Context, vectorStoreID, fileID string) (*VectorStoreFile, error) {
	payload := map[string]interface{}{
		"file_id": fileID,
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal attach file params: %w", err)
	}

	url := fmt.Sprintf("%s/v1/openai/v1/vector_stores/%s/files", c.BaseURL, vectorStoreID)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.APIKey)

	fmt.Println("=== REST CALL: Attach File to Vector Store ===")
	fmt.Printf("URL: %s\n", url)
	fmt.Printf("Method: %s\n", req.Method)
	fmt.Printf("Headers: %v\n", req.Header)
	fmt.Printf("Request Body:\n%s\n", string(jsonData))

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	fmt.Printf("Response Status: %s\n", resp.Status)
	fmt.Printf("Response Headers: %v\n", resp.Header)

	body, _ := io.ReadAll(resp.Body)
	fmt.Printf("Response Body:\n%s\n", string(body))
	fmt.Println("=== END REST CALL ===\n")

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var response VectorStoreFile
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &response, nil
}

// InsertDocumentsIntoRAG inserts documents into the RAG system
func (c *LlamaStackClient) InsertDocumentsIntoRAG(ctx context.Context, params RagToolInsertParams) error {
	jsonData, err := json.Marshal(params)
	if err != nil {
		return fmt.Errorf("failed to marshal RAG insert params: %w", err)
	}

	url := c.BaseURL + "/v1/tool-runtime/rag-tool/insert"
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.APIKey)
	req.Header.Set("Accept", "*/*")

	fmt.Println("=== REST CALL: Insert Documents into RAG ===")
	fmt.Printf("URL: %s\n", url)
	fmt.Printf("Method: %s\n", req.Method)
	fmt.Printf("Headers: %v\n", req.Header)
	fmt.Printf("Request Body:\n%s\n", string(jsonData))

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	fmt.Printf("Response Status: %s\n", resp.Status)
	fmt.Printf("Response Headers: %v\n", resp.Header)

	body, _ := io.ReadAll(resp.Body)
	fmt.Printf("Response Body:\n%s\n", string(body))
	fmt.Println("=== END REST CALL ===\n")

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	return nil
}

// CreateAgent creates a new agent
func (c *LlamaStackClient) CreateAgent(ctx context.Context, params AgentCreateParams) (*APIResponse, error) {
	jsonData, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal agent params: %w", err)
	}

	url := c.BaseURL + "/v1/agents"
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.APIKey)

	fmt.Println("=== REST CALL: Create Agent ===")
	fmt.Printf("URL: %s\n", url)
	fmt.Printf("Method: %s\n", req.Method)
	fmt.Printf("Headers: %v\n", req.Header)
	fmt.Printf("Request Body:\n%s\n", string(jsonData))

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	fmt.Printf("Response Status: %s\n", resp.Status)
	fmt.Printf("Response Headers: %v\n", resp.Header)

	body, _ := io.ReadAll(resp.Body)
	fmt.Printf("Response Body:\n%s\n", string(body))
	fmt.Println("=== END REST CALL ===\n")

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var response APIResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &response, nil
}

// DeleteAgent deletes an agent by ID
func (c *LlamaStackClient) DeleteAgent(ctx context.Context, agentID string) error {
	req, err := http.NewRequestWithContext(ctx, "DELETE", c.BaseURL+"/v1/agents/"+agentID, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+c.APIKey)
	req.Header.Set("Accept", "*/*")

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusNoContent {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	return nil
}

// CreateChatCompletion creates a chat completion (non-streaming)
func (c *LlamaStackClient) CreateChatCompletion(ctx context.Context, params ChatCompletionParams) (*APIResponse, error) {
	jsonData, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal chat completion params: %w", err)
	}

	url := c.BaseURL + "/v1/openai/v1/chat/completions"
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.APIKey)

	fmt.Println("=== REST CALL: Create Chat Completion ===")
	fmt.Printf("URL: %s\n", url)
	fmt.Printf("Method: %s\n", req.Method)
	fmt.Printf("Headers: %v\n", req.Header)
	fmt.Printf("Request Body:\n%s\n", string(jsonData))

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	fmt.Printf("Response Status: %s\n", resp.Status)
	fmt.Printf("Response Headers: %v\n", resp.Header)

	body, _ := io.ReadAll(resp.Body)
	fmt.Printf("Response Body:\n%s\n", string(body))
	fmt.Println("=== END REST CALL ===\n")

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var response APIResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &response, nil
}

// CreateStreamingChatCompletion creates a streaming chat completion
func (c *LlamaStackClient) CreateStreamingChatCompletion(ctx context.Context, params ChatCompletionParams) (<-chan string, error) {
	// Set streaming to true
	stream := true
	params.Stream = &stream

	jsonData, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal chat completion params: %w", err)
	}

	url := c.BaseURL + "/v1/openai/v1/chat/completions"
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.APIKey)

	fmt.Println("=== REST CALL: Create Streaming Chat Completion ===")
	fmt.Printf("URL: %s\n", url)
	fmt.Printf("Method: %s\n", req.Method)
	fmt.Printf("Headers: %v\n", req.Header)
	fmt.Printf("Request Body:\n%s\n", string(jsonData))

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %w", err)
	}

	fmt.Printf("Response Status: %s\n", resp.Status)
	fmt.Printf("Response Headers: %v\n", resp.Header)
	fmt.Println("=== END REST CALL ===\n")

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	// Create channel for streaming responses
	ch := make(chan string)

	go func() {
		defer resp.Body.Close()
		defer close(ch)

		reader := bufio.NewReader(resp.Body)
		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				if err == io.EOF {
					break
				}
				ch <- fmt.Sprintf("Error reading stream: %v", err)
				return
			}

			// Skip empty lines
			if line == "\n" {
				continue
			}

			// Remove "data: " prefix if present
			if len(line) > 6 && line[:6] == "data: " {
				line = line[6:]
			}

			// Check for end of stream
			if line == "[DONE]\n" {
				break
			}

			ch <- line
		}
	}()

	return ch, nil
}

// Model represents a model from the API
type Model struct {
	Identifier string `json:"identifier"`
	ModelType  string `json:"model_type"`
	Name       string `json:"name,omitempty"`
}

// ListModelsResponse represents the response from listing models
type ListModelsResponse struct {
	Data []Model `json:"data"`
}

// ListModels lists available models
func (c *LlamaStackClient) ListModels(ctx context.Context) (*ListModelsResponse, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", c.BaseURL+"/v1/models", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+c.APIKey)

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var response ListModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &response, nil
}

// GetAvailableModel gets the first available LLM model
func (c *LlamaStackClient) GetAvailableModel(ctx context.Context) (string, error) {
	models, err := c.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to list models: %w", err)
	}

	// Filter for LLM models like the TypeScript example
	var availableModels []string
	for _, model := range models.Data {
		if model.ModelType == "llm" &&
			!strings.Contains(model.Identifier, "guard") &&
			!strings.Contains(model.Identifier, "405") {
			availableModels = append(availableModels, model.Identifier)
		}
	}

	if len(availableModels) == 0 {
		return "", fmt.Errorf("no available models found")
	}

	return availableModels[0], nil
}

// Session represents a session
type Session struct {
	SessionID   string `json:"session_id"`
	AgentID     string `json:"agent_id"`
	SessionName string `json:"session_name"`
	CreatedAt   int64  `json:"created_at"`
}

// SessionCreateParams represents parameters for creating a session
type SessionCreateParams struct {
	SessionName string `json:"session_name"`
}

// Turn represents a turn in an agent session
type Turn struct {
	TurnID            string        `json:"turn_id"`
	SessionID         string        `json:"session_id"`
	InputMessages     []Message     `json:"input_messages"`
	OutputMessage     Message       `json:"output_message"`
	Steps             []interface{} `json:"steps"`
	StartedAt         string        `json:"started_at"`
	CompletedAt       *string       `json:"completed_at,omitempty"`
	OutputAttachments []interface{} `json:"output_attachments,omitempty"`
}

// TurnCreateParams represents parameters for creating a turn
type TurnCreateParams struct {
	Messages   []Message  `json:"messages"`
	Stream     *bool      `json:"stream,omitempty"`
	Documents  []Document `json:"documents,omitempty"`
	ToolConfig *struct {
		ToolChoice string `json:"tool_choice,omitempty"`
	} `json:"tool_config,omitempty"`
	Toolgroups []interface{} `json:"toolgroups,omitempty"`
}

// RagToolQueryParams represents parameters for RAG tool query
type RagToolQueryParams struct {
	Content     string   `json:"content"`
	VectorDBIDs []string `json:"vector_db_ids"`
	QueryConfig *struct {
		MaxChunks          int    `json:"max_chunks"`
		MaxTokensInContext int    `json:"max_tokens_in_context"`
		Mode               string `json:"mode"`
	} `json:"query_config,omitempty"`
}

// QueryResult represents the result of a RAG query
type QueryResult struct {
	Content  []interface{}          `json:"content"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// CreateSession creates a new session for an agent
func (c *LlamaStackClient) CreateSession(ctx context.Context, agentID string, params SessionCreateParams) (*Session, error) {
	jsonData, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal session params: %w", err)
	}

	url := fmt.Sprintf("%s/v1/agents/%s/session", c.BaseURL, agentID)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.APIKey)

	fmt.Println("=== REST CALL: Create Session ===")
	fmt.Printf("URL: %s\n", url)
	fmt.Printf("Method: %s\n", req.Method)
	fmt.Printf("Headers: %v\n", req.Header)
	fmt.Printf("Request Body:\n%s\n", string(jsonData))

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	fmt.Printf("Response Status: %s\n", resp.Status)
	fmt.Printf("Response Headers: %v\n", resp.Header)

	body, _ := io.ReadAll(resp.Body)
	fmt.Printf("Response Body:\n%s\n", string(body))
	fmt.Println("=== END REST CALL ===\n")

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var response Session
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &response, nil
}

// CreateTurn creates a new turn for an agent session (supports streaming SSE)
func (c *LlamaStackClient) CreateTurn(ctx context.Context, agentID, sessionID string, params TurnCreateParams) (*Turn, error) {
	jsonData, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal turn params: %w", err)
	}

	url := fmt.Sprintf("%s/v1/agents/%s/session/%s/turn", c.BaseURL, agentID, sessionID)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.APIKey)

	fmt.Println("=== REST CALL: Create Turn (Streaming) ===")
	fmt.Printf("URL: %s\n", url)
	fmt.Printf("Method: %s\n", req.Method)
	fmt.Printf("Headers: %v\n", req.Header)
	fmt.Printf("Request Body:\n%s\n", string(jsonData))

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %w", err)
	}
	// Do not defer resp.Body.Close() here, as we need to stream

	fmt.Printf("Response Status: %s\n", resp.Status)
	fmt.Printf("Response Headers: %v\n", resp.Header)

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	// Parse SSE events
	turn, err := parseAgentTurnSSE(resp.Body)
	resp.Body.Close()
	if err != nil {
		return nil, fmt.Errorf("failed to parse SSE: %w", err)
	}

	return turn, nil
}

// parseAgentTurnSSE parses the SSE stream and returns the final Turn when turn_complete is received
func parseAgentTurnSSE(body io.Reader) (*Turn, error) {
	scanner := bufio.NewScanner(body)
	var turn Turn
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			jsonPart := strings.TrimPrefix(line, "data: ")
			var sse struct {
				Event struct {
					Payload struct {
						EventType string `json:"event_type"`
						Turn      *Turn  `json:"turn,omitempty"`
						// For step_progress, etc, you could add more fields if needed
					} `json:"payload"`
				} `json:"event"`
			}
			err := json.Unmarshal([]byte(jsonPart), &sse)
			if err != nil {
				fmt.Printf("[SSE] Failed to parse event: %v\n", err)
				continue
			}
			if sse.Event.Payload.EventType == "turn_complete" && sse.Event.Payload.Turn != nil {
				turn = *sse.Event.Payload.Turn
				break
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scanner error: %w", err)
	}
	if turn.TurnID == "" {
		return nil, fmt.Errorf("no turn_complete event received")
	}
	return &turn, nil
}

// QueryRAG queries the RAG system for context
func (c *LlamaStackClient) QueryRAG(ctx context.Context, params RagToolQueryParams) (*QueryResult, error) {
	jsonData, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal RAG query params: %w", err)
	}

	url := c.BaseURL + "/v1/tool-runtime/rag-tool/query"
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.APIKey)

	fmt.Println("=== REST CALL: Query RAG ===")
	fmt.Printf("URL: %s\n", url)
	fmt.Printf("Method: %s\n", req.Method)
	fmt.Printf("Headers: %v\n", req.Header)
	fmt.Printf("Request Body:\n%s\n", string(jsonData))

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	fmt.Printf("Response Status: %s\n", resp.Status)
	fmt.Printf("Response Headers: %v\n", resp.Header)

	body, _ := io.ReadAll(resp.Body)
	fmt.Printf("Response Body:\n%s\n", string(body))
	fmt.Println("=== END REST CALL ===\n")

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var response QueryResult
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &response, nil
}

// ListFilesResponse represents the response from listing files
type ListFilesResponse struct {
	Data    []FileResponse `json:"data"`
	FirstID string         `json:"first_id"`
	HasMore bool           `json:"has_more"`
	LastID  string         `json:"last_id"`
	Object  string         `json:"object"`
}

// ListFiles lists uploaded files
func (c *LlamaStackClient) ListFiles(ctx context.Context) (*ListFilesResponse, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", c.BaseURL+"/v1/openai/v1/files", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+c.APIKey)

	fmt.Println("=== REST CALL: List Files ===")
	fmt.Printf("URL: %s\n", req.URL)
	fmt.Printf("Method: %s\n", req.Method)
	fmt.Printf("Headers: %v\n", req.Header)

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	fmt.Printf("Response Status: %s\n", resp.Status)
	fmt.Printf("Response Headers: %v\n", resp.Header)

	body, _ := io.ReadAll(resp.Body)
	fmt.Printf("Response Body:\n%s\n", string(body))
	fmt.Println("=== END REST CALL ===\n")

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var response ListFilesResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &response, nil
}

// Example usage functions
func exampleCreateAgent(client *LlamaStackClient) {
	ctx := context.Background()

	selectedModel := "ollama/llama3.2:3b"
	fmt.Printf("Using model: %s\n", selectedModel)

	// Create agent configuration with required instructions and fields matching TypeScript example
	temperature := 1.0
	topP := 0.9
	maxInferIters := 10

	agentConfig := AgentConfig{
		Instructions: "You are a helpful assistant",
		Model:        selectedModel,
		Name:         "Example Agent",
		Description:  "A sample agent for demonstration",
		SamplingParams: &SamplingParams{
			Strategy: SamplingStrategy{
				Type:        "top_p",
				Temperature: &temperature,
				TopP:        &topP,
			},
		},
		ToolChoice:               "auto",
		ToolPromptFormat:         "python_list",
		InputShields:             []string{},
		OutputShields:            []string{},
		EnableSessionPersistence: false,
		MaxInferIters:            maxInferIters,
		Toolgroups:               []interface{}{},
		Tools: []map[string]interface{}{
			{
				"type": "function",
				"function": map[string]interface{}{
					"name":        "get_weather",
					"description": "Get weather information for a location",
					"parameters": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"location": map[string]interface{}{
								"type":        "string",
								"description": "The location to get weather for",
							},
						},
						"required": []string{"location"},
					},
				},
			},
		},
	}

	params := AgentCreateParams{
		AgentConfig: agentConfig,
	}

	// Debug: Print the JSON payload to match TypeScript example
	jsonData, _ := json.MarshalIndent(params, "", "  ")
	fmt.Println("Agent Configuration Payload:")
	fmt.Println(string(jsonData))
	fmt.Println()

	response, err := client.CreateAgent(ctx, params)
	if err != nil {
		fmt.Printf("Error creating agent: %v\n", err)
		return
	}

	fmt.Printf("Agent created successfully! Agent ID: %s\n", response.AgentID)
}

func exampleChatCompletion(client *LlamaStackClient, userPrompt string) {
	ctx := context.Background()

	selectedModel := "ollama/llama3.2:3b"
	fmt.Printf("Using model: %s\n", selectedModel)

	// Create chat completion parameters
	params := ChatCompletionParams{
		Model: selectedModel,
		Messages: []Message{
			{
				Role:    "system",
				Content: "You are a helpful assistant.",
			},
			{
				Role:    "user",
				Content: userPrompt,
			},
		},
	}

	response, err := client.CreateChatCompletion(ctx, params)
	if err != nil {
		fmt.Printf("Error creating chat completion: %v\n", err)
		return
	}

	// Extract and display just the message content (like TypeScript client)
	if len(response.Choices) > 0 {
		messageContent := response.Choices[0].Message.Content
		fmt.Printf("Response: %s\n", messageContent)
	} else {
		fmt.Println("No response content received")
	}
}

func exampleStreamingChatCompletion(client *LlamaStackClient, userPrompt string) {
	ctx := context.Background()

	selectedModel := "ollama/llama3.2:3b"
	fmt.Printf("Using model: %s\n", selectedModel)

	// Create streaming chat completion parameters
	params := ChatCompletionParams{
		Model: selectedModel,
		Messages: []Message{
			{
				Role:    "system",
				Content: "You are a helpful assistant.",
			},
			{
				Role:    "user",
				Content: userPrompt,
			},
		},
	}

	stream, err := client.CreateStreamingChatCompletion(ctx, params)
	if err != nil {
		fmt.Printf("Error creating streaming chat completion: %v\n", err)
		return
	}

	fmt.Println("Streaming response:")
	for chunk := range stream {
		fmt.Print(chunk)
	}
	fmt.Println()
}

// New function: Example PDF upload and RAG workflow
func examplePDFUploadAndRAG(client *LlamaStackClient, pdfPath string) {
	ctx := context.Background()

	fmt.Println("=== PDF Upload and RAG Workflow ===")

	// Step 1: Upload the PDF file
	fmt.Println("Step 1: Uploading PDF file...")
	fileResponse, err := client.UploadFile(ctx, pdfPath, "assistants")
	if err != nil {
		fmt.Printf("Error uploading file: %v\n", err)
		return
	}
	fmt.Printf("File uploaded successfully! File ID: %s\n", fileResponse.ID)

	// Step 2: Create a vector store
	fmt.Println("Step 2: Creating vector store...")
	vectorStore, err := client.CreateVectorStore(ctx, "my-documents", map[string]interface{}{
		"description": "Vector store for PDF documents",
		"source":      "go-client",
	})
	if err != nil {
		fmt.Printf("Error creating vector store: %v\n", err)
		return
	}
	fmt.Printf("Vector store created successfully! Vector Store ID: %s\n", vectorStore.ID)

	// Step 3: Attach the file to the vector store
	fmt.Println("Step 3: Attaching file to vector store...")
	vectorStoreFile, err := client.AttachFileToVectorStore(ctx, vectorStore.ID, fileResponse.ID)
	if err != nil {
		fmt.Printf("Error attaching file to vector store: %v\n", err)
		return
	}
	fmt.Printf("File attached successfully! Status: %s\n", vectorStoreFile.Status)

	// Step 4: Insert documents into RAG system (alternative approach)
	fmt.Println("Step 4: Inserting documents into RAG system...")

	// Read the PDF content (simplified - in real scenario you'd extract text from PDF)
	pdfContent := "Eder dog is Bella, a Cavalier King breed. Ana dog is Dora, a Pug breed."

	ragParams := RagToolInsertParams{
		ChunkSizeInTokens: 1000,
		Documents: []Document{
			{
				Content:    pdfContent,
				DocumentID: "sample-pdf-doc",
				Metadata: map[string]interface{}{
					"source":      "sample.pdf",
					"type":        "pdf",
					"uploaded_by": "go-client",
				},
				MimeType: "application/pdf",
			},
		},
		VectorDBID: vectorStore.ID,
	}

	err = client.InsertDocumentsIntoRAG(ctx, ragParams)
	if err != nil {
		fmt.Printf("Error inserting documents into RAG: %v\n", err)
		return
	}
	fmt.Println("Documents inserted into RAG system successfully!")

	fmt.Println("=== PDF Upload and RAG Workflow Completed ===")
}

// New function: Agent-based chat with RAG
func exampleAgentChatWithRAG(client *LlamaStackClient, userPrompt string) {
	ctx := context.Background()

	fmt.Println("=== Agent Chat with RAG (Agentic Loop) ===")

	selectedModel := "ollama/llama3.2:3b"
	fmt.Printf("Using model: %s\n", selectedModel)

	// Step 1: Create an agent with RAG toolgroups
	fmt.Println("Step 1: Creating agent with RAG capabilities...")
	temperature := 1.0
	topP := 0.9
	maxInferIters := 10

	agentConfig := AgentConfig{
		Instructions: "You are a helpful assistant that can access documents through RAG tools. When asked about documents, use the RAG tools to find relevant information.",
		Model:        selectedModel,
		Name:         "RAG Agent",
		Description:  "An agent with RAG capabilities",
		SamplingParams: &SamplingParams{
			Strategy: SamplingStrategy{
				Type:        "top_p",
				Temperature: &temperature,
				TopP:        &topP,
			},
		},
		ToolChoice:               "auto",
		ToolPromptFormat:         "python_list",
		InputShields:             []string{},
		OutputShields:            []string{},
		EnableSessionPersistence: false,
		MaxInferIters:            maxInferIters,
		Toolgroups: []interface{}{
			map[string]interface{}{
				"name": "builtin::rag",
				"args": map[string]interface{}{
					"vector_db_ids": []string{"my-documents"},
				},
			},
		},
	}

	params := AgentCreateParams{
		AgentConfig: agentConfig,
	}

	response, err := client.CreateAgent(ctx, params)
	if err != nil {
		fmt.Printf("Error creating agent: %v\n", err)
		return
	}

	agentID := response.AgentID
	fmt.Printf("Agent created successfully! Agent ID: %s\n", agentID)

	// Step 2: Create a session
	fmt.Println("Step 2: Creating session...")
	sessionParams := SessionCreateParams{
		SessionName: "pdf-chat-session",
	}

	session, err := client.CreateSession(ctx, agentID, sessionParams)
	if err != nil {
		fmt.Printf("Error creating session: %v\n", err)
		return
	}

	sessionID := session.SessionID
	fmt.Printf("Session created successfully! Session ID: %s\n", sessionID)

	// Step 3: Create a turn with the user prompt (streaming)
	fmt.Println("Step 3: Creating turn with user prompt (streaming)...")
	// turnParams := TurnCreateParams{ ... } // REMOVE this line, now handled in initParams

	// Start the agentic loop
	agenticLoop := func(agentID, sessionID string, turnParams map[string]interface{}) {
		turnID := ""
		for {
			url := fmt.Sprintf("%s/v1/agents/%s/session/%s/turn", client.BaseURL, agentID, sessionID)
			if turnID != "" {
				url = fmt.Sprintf("%s/v1/agents/%s/session/%s/turn/%s/resume", client.BaseURL, agentID, sessionID, turnID)
			}

			jsonData, err := json.Marshal(turnParams)
			if err != nil {
				fmt.Printf("Failed to marshal turn params: %v\n", err)
				return
			}

			method := "POST"
			req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewBuffer(jsonData))
			if err != nil {
				fmt.Printf("Failed to create request: %v\n", err)
				return
			}
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Authorization", "Bearer "+client.APIKey)

			fmt.Println("=== REST CALL: Agent Turn (Streaming) ===")
			fmt.Printf("URL: %s\n", url)
			fmt.Printf("Method: %s\n", req.Method)
			fmt.Printf("Headers: %v\n", req.Header)
			fmt.Printf("Request Body:\n%s\n", string(jsonData))

			resp, err := client.HTTPClient.Do(req)
			if err != nil {
				fmt.Printf("Failed to make request: %v\n", err)
				return
			}
			defer resp.Body.Close()

			scanner := bufio.NewScanner(resp.Body)
			var awaitingInputTurn *Turn
			var finalTurn *Turn
			for scanner.Scan() {
				line := scanner.Text()
				if strings.HasPrefix(line, "data: ") {
					jsonPart := strings.TrimPrefix(line, "data: ")
					fmt.Printf("[DEBUG] SSE Event: %s\n", jsonPart)
					var sse struct {
						Event struct {
							Payload struct {
								EventType string `json:"event_type"`
								Turn      *Turn  `json:"turn,omitempty"`
							} `json:"payload"`
						} `json:"event"`
					}
					err := json.Unmarshal([]byte(jsonPart), &sse)
					if err != nil {
						fmt.Printf("[SSE] Failed to parse event: %v\n", err)
						continue
					}
					eventType := sse.Event.Payload.EventType
					fmt.Printf("[DEBUG] Event type: %s\n", eventType)
					if eventType == "turn_complete" && sse.Event.Payload.Turn != nil {
						finalTurn = sse.Event.Payload.Turn
						fmt.Printf("[DEBUG] Turn complete received\n")
						break
					} else if eventType == "turn_awaiting_input" && sse.Event.Payload.Turn != nil {
						awaitingInputTurn = sse.Event.Payload.Turn
						fmt.Printf("[DEBUG] Turn awaiting input received\n")
						break
					}
				}
			}
			if err := scanner.Err(); err != nil {
				fmt.Printf("Scanner error: %v\n", err)
				return
			}

			if finalTurn != nil {
				fmt.Printf("\n=== Agent Final Response ===\n%s\n", finalTurn.OutputMessage.Content)
				fmt.Println("=== Agent Chat with RAG Completed ===")
				return
			}

			if awaitingInputTurn != nil {
				// Find tool calls in steps
				var toolCalls []struct {
					CallID    string
					ToolName  string
					Arguments interface{}
				}
				for _, step := range awaitingInputTurn.Steps {
					stepMap, ok := step.(map[string]interface{})
					if !ok {
						continue
					}
					if stepMap["step_type"] == "tool_execution" {
						calls, ok := stepMap["tool_calls"].([]interface{})
						if !ok {
							continue
						}
						for _, call := range calls {
							callMap, ok := call.(map[string]interface{})
							if !ok {
								continue
							}
							callID, _ := callMap["call_id"].(string)
							toolName, _ := callMap["tool_name"].(string)
							arguments := callMap["arguments"]
							toolCalls = append(toolCalls, struct {
								CallID    string
								ToolName  string
								Arguments interface{}
							}{
								CallID:    callID,
								ToolName:  toolName,
								Arguments: arguments,
							})
						}
					}
				}
				if len(toolCalls) == 0 {
					fmt.Println("No tool calls found in awaiting_input turn.")
					return
				}
				// For each tool call, handle RAG
				var toolResponses []map[string]interface{}
				for _, call := range toolCalls {
					if strings.Contains(call.ToolName, "rag") || strings.Contains(call.ToolName, "knowledge_search") {
						// Assume arguments is a string or map with 'query' or 'content'
						var query string
						switch v := call.Arguments.(type) {
						case string:
							query = v
						case map[string]interface{}:
							if q, ok := v["query"].(string); ok {
								query = q
							} else if c, ok := v["content"].(string); ok {
								query = c
							}
						}
						if query == "" {
							fmt.Printf("No query found in tool call arguments: %+v\n", call.Arguments)
							continue
						}
						// Call RAG
						ragParams := RagToolQueryParams{
							Content:     query,
							VectorDBIDs: []string{"my-documents"}, // TODO: make dynamic if needed
						}
						ragResult, err := client.QueryRAG(ctx, ragParams)
						if err != nil {
							fmt.Printf("Error querying RAG: %v\n", err)
							continue
						}
						// Compose tool response
						var ragText string
						if len(ragResult.Content) > 0 {
							if itemMap, ok := ragResult.Content[0].(map[string]interface{}); ok {
								if text, exists := itemMap["text"].(string); exists {
									ragText = text
								}
							}
						}
						if ragText == "" {
							ragText = "[No relevant context found in RAG]"
						}
						toolResponses = append(toolResponses, map[string]interface{}{
							"call_id":   call.CallID,
							"tool_name": call.ToolName,
							"content": map[string]interface{}{
								"type": "text",
								"text": ragText,
							},
						})
					} else {
						fmt.Printf("Skipping non-RAG tool call: %s\n", call.ToolName)
					}
				}
				if len(toolResponses) == 0 {
					fmt.Println("No tool responses to send.")
					return
				}
				// Resume the turn with tool responses
				resumeParams := map[string]interface{}{
					"tool_responses": toolResponses,
					"stream":         true,
				}
				// Next loop iteration will POST to /turn/{turn_id}/resume
				turnID = awaitingInputTurn.TurnID
				turnParams = resumeParams
				continue
			}
			fmt.Println("No final or awaiting_input turn received.")
			return
		}
	}

	// Prepare initial turnParams as map[string]interface{}
	initParams := map[string]interface{}{
		"messages": []Message{
			{
				Role:    "user",
				Content: userPrompt,
			},
		},
		"stream": true,
	}

	agenticLoop(agentID, sessionID, initParams)
}

// New function: Direct RAG query
func exampleDirectRAGQuery(client *LlamaStackClient, userPrompt string) {
	ctx := context.Background()

	fmt.Println("=== Direct RAG Query ===")

	// Query the RAG system directly
	queryParams := RagToolQueryParams{
		Content:     userPrompt,
		VectorDBIDs: []string{"my-documents"}, // Use the vector store we created
		QueryConfig: &struct {
			MaxChunks          int    `json:"max_chunks"`
			MaxTokensInContext int    `json:"max_tokens_in_context"`
			Mode               string `json:"mode"`
		}{
			MaxChunks:          5,
			MaxTokensInContext: 1000,
			Mode:               "vector",
		},
	}

	result, err := client.QueryRAG(ctx, queryParams)
	if err != nil {
		fmt.Printf("Error querying RAG: %v\n", err)
		return
	}

	fmt.Printf("RAG Query Result:\n")
	for i, item := range result.Content {
		if itemMap, ok := item.(map[string]interface{}); ok {
			if text, exists := itemMap["text"]; exists {
				fmt.Printf("Item %d: %s\n", i+1, text)
			}
		}
	}
	fmt.Println("=== Direct RAG Query Completed ===")
}

// New function: Chat completion with PDF context
func exampleChatCompletionWithPDF(client *LlamaStackClient, userPrompt string) {
	ctx := context.Background()

	selectedModel := "ollama/llama3.2:3b"
	fmt.Printf("Using model: %s\n", selectedModel)

	// Create chat completion parameters with context about the uploaded PDF
	systemPrompt := "You have access to a PDF document that was uploaded. Please answer questions based on the content of that document. If the question is not related to the document, you can provide a general helpful response."

	params := ChatCompletionParams{
		Model: selectedModel,
		Messages: []Message{
			{
				Role:    "system",
				Content: systemPrompt,
			},
			{
				Role:    "user",
				Content: userPrompt,
			},
		},
	}

	response, err := client.CreateChatCompletion(ctx, params)
	if err != nil {
		fmt.Printf("Error creating chat completion: %v\n", err)
		return
	}

	// Extract and display just the message content (like TypeScript client)
	if len(response.Choices) > 0 {
		messageContent := response.Choices[0].Message.Content
		fmt.Printf("Response: %s\n", messageContent)
	} else {
		fmt.Println("No response content received")
	}
}

// New function: List uploaded files
func exampleListFiles(client *LlamaStackClient) {
	ctx := context.Background()

	fmt.Println("=== List Uploaded Files ===")

	files, err := client.ListFiles(ctx)
	if err != nil {
		fmt.Printf("Error listing files: %v\n", err)
		return
	}

	fmt.Printf("Found %d uploaded files:\n", len(files.Data))
	for i, file := range files.Data {
		fmt.Printf("  %d. ID: %s, Filename: %s, Size: %d bytes, Purpose: %s, Created: %d\n",
			i+1, file.ID, file.Filename, file.Bytes, file.Purpose, file.CreatedAt)
	}
	fmt.Println("=== List Files Completed ===")
}

func main() {
	// Check for command line arguments
	var userPrompt string
	var pdfPath string

	if len(os.Args) > 1 {
		userPrompt = os.Args[1]
	} else {
		userPrompt = "Who is Dora's owner?" // default prompt
	}

	// Check for PDF file path argument
	if len(os.Args) > 2 {
		pdfPath = os.Args[2]
	} else {
		pdfPath = "sample.pdf" // default PDF path
	}

	// Initialize the client
	// Use localhost like the TypeScript examples
	baseURL := "http://localhost:8321"
	apiKey := "your-api-key-here"

	client := NewLlamaStackClient(baseURL, apiKey)

	fmt.Println("=== Llama Stack API Go Sample ===")
	fmt.Printf("Using base URL: %s\n", baseURL)
	fmt.Printf("User prompt: %s\n", userPrompt)
	fmt.Printf("PDF file path: %s\n", pdfPath)
	fmt.Println()

	// Only run the PDF upload and agentic RAG test for debugging
	fmt.Println("1. PDF Upload and RAG workflow...")
	examplePDFUploadAndRAG(client, pdfPath)
	fmt.Println()

	fmt.Println("2. Agent-based chat with RAG...")
	exampleAgentChatWithRAG(client, userPrompt)
	fmt.Println()

	// List files first to see what's already uploaded
	fmt.Println("0. List uploaded files...")
	// exampleListFiles(client)
	fmt.Println()

	fmt.Println("Sample completed!")
}
