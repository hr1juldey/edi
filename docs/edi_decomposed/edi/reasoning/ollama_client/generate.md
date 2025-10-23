# OllamaClient.generate()

[Back to Ollama Client](../reasoning_ollama_client.md)

## Related User Story
"As a user, I want EDI to understand my intent and generate appropriate editing instructions." (from PRD - core functionality)

## Function Signature
`generate(prompt, model) -> str`

## Parameters
- `prompt` - The input prompt to send to the LLM
- `model` - The identifier for the model to use (e.g., "qwen3:8b", "gemma3:4b")

## Returns
- `str` - The generated text response from the LLM

## Step-by-step Logic
1. Prepare the API request with the given prompt and model parameters
2. Send the request to the Ollama server endpoint
3. Handle the response from the Ollama API
4. Process any streaming responses if applicable
5. Handle connection errors, timeouts, and retry if needed
6. Return the generated text response
7. Log the interaction for debugging and learning purposes

## Error Handling
- Handles network connection issues with retries
- Manages timeout scenarios gracefully
- Validates response from Ollama server
- Provides fallback mechanisms if primary model fails
- Reports clear error messages to higher-level functions

## Performance Optimizations
- Uses streaming API to show partial results
- Sets appropriate context size (num_ctx=4096) for faster processing
- Maintains connection to Ollama server to avoid cold starts
- Implements efficient request/response processing

## Input/Output Data Structures
### Input
- Prompt: Text string containing the instruction for the LLM
- Model: String identifier for the specific model to use

### Output
- Generated text response from the LLM as a string
- May contain structured data (JSON) depending on the prompt