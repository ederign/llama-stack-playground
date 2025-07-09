package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
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

func main() {
	// Check for command line arguments
	var userPrompt string
	if len(os.Args) > 1 {
		userPrompt = os.Args[1]
	} else {
		userPrompt = "How are you?" // default prompt
	}

	// Initialize the client
	// Use localhost like the TypeScript examples
	baseURL := "http://localhost:8321"
	apiKey := "your-api-key-here"

	client := NewLlamaStackClient(baseURL, apiKey)

	fmt.Println("=== Llama Stack API Go Sample ===")
	fmt.Printf("Using base URL: %s\n", baseURL)
	fmt.Printf("User prompt: %s\n", userPrompt)
	fmt.Println()

	// Example 1: Create an agent
	fmt.Println("1. Creating an agent...")
	exampleCreateAgent(client)
	fmt.Println()

	// Example 2: Create a chat completion with user prompt
	fmt.Println("2. Creating a chat completion...")
	exampleChatCompletion(client, userPrompt)
	fmt.Println()

	// Example 3: Create a streaming chat completion with user prompt
	fmt.Println("3. Creating a streaming chat completion...")
	//exampleStreamingChatCompletion(client, userPrompt)
	fmt.Println()

	fmt.Println("Sample completed!")
}
