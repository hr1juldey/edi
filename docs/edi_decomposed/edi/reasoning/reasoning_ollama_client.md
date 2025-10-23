# Reasoning: Ollama Client

[Back to Reasoning Subsystem](./reasoning_subsystem.md)

## Purpose
Ollama API wrapper - Contains the OllamaClient class that handles communication with Ollama for LLM inference.

## Class: OllamaClient

### Methods
- `generate(prompt, model) -> str`: Sends a prompt to the specified model and returns the generated text

### Details
- Handles connection errors and retries
- Manages communication with Ollama server
- Provides a clean interface to Ollama's API

## Functions

- [generate(prompt, model)](./reasoning/generate.md)

## Technology Stack

- Requests for HTTP communication
- Ollama for LLM inference