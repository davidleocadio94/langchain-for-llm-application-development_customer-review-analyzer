---
title: Customer Review Analyzer
emoji: 📝
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 5.16.0
app_file: app.py
pinned: false
---

# Customer Review Analyzer

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-blue)](https://huggingface.co/spaces/davidleocadio94DLAI/langchain-for-llm-application-development_customer-review-analyzer)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-Chains-orange)](https://langchain.com)

Analyze customer reviews using LangChain with structured output parsers, sequential chains, and conversation memory. Supports multiple languages and generates professional customer service responses.

## Features

- **Structured Analysis** - Extract sentiment, summary, key issues, and urgency level
- **Multi-Language Support** - Detects language and responds in the same language
- **Sequential Chains** - Pipeline: analyze → detect language → generate response
- **Conversation Memory** - Interactive chat mode remembers context

## Tech Stack

![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5--Turbo-412991)
![LangChain](https://img.shields.io/badge/LangChain-Chains%20%26%20Memory-1C3C3C)
![Pydantic](https://img.shields.io/badge/Structured-Output%20Parsers-E92063)
![Gradio](https://img.shields.io/badge/Gradio-UI-F97316)

## Getting Started

### Prerequisites

- Python 3.10+
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/davidleocadio94/langchain-for-llm-application-development_customer-review-analyzer.git
cd langchain-for-llm-application-development_customer-review-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Run Locally

```bash
# Set your API key
export OPENAI_API_KEY=your-api-key

# Run the app
python app.py
```

Open http://localhost:7860 in your browser.

### Run with Docker

```bash
# Build the image
docker build -t customer-review-analyzer .

# Run the container
docker run -p 7860:7860 -e OPENAI_API_KEY=your-api-key customer-review-analyzer
```

Open http://localhost:7860 in your browser.

## Three Analysis Modes

### 1. Quick Analysis
Extract structured information from a review:
- **Sentiment**: positive, negative, or neutral
- **Summary**: One-sentence summary
- **Key Issues**: List of problems mentioned
- **Urgency**: Whether immediate help is needed

### 2. Full Pipeline
Complete sequential chain processing:
1. Analyze the review (sentiment, issues, urgency)
2. Detect the language
3. Generate a professional response in that language

### 3. Interactive Chat
Conversational mode with memory:
- Share reviews and ask follow-up questions
- Context is maintained across messages
- Clear memory to start fresh

## Example Reviews

- English (Negative): "This product is terrible! It broke after one day..."
- French (Positive): "J'adore ce produit! La qualité est exceptionnelle..."
- Spanish (Positive): "¡Excelente servicio! El equipo de soporte fue muy amable..."

---

Built as part of the [LangChain for LLM Application Development](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/) course on DeepLearning.AI
