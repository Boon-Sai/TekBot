# Problem Statement: TekBot – Rule-Based Company Chatbot
## Context
In modern digital-first organizations, users and potential candidates often find it difficult to navigate through complex corporate websites to access specific information such as services offered, job opportunities, work culture, leadership details, or company growth.

Tekworks, being a dynamic executive hiring company, wants to simplify user experience on their site by introducing a conversational interface that serves as a one-stop assistant for all company-related queries.

## Objective
The goal is to develop a rule-based chatbot named “TekBot” that uses Natural Language Processing (NLP) and Machine Learning to:

- Identify the intent of a user query (e.g., asking about services, jobs, company background, etc.)

- Respond with accurate, pre-defined answers based on a labeled dataset

- Provide a streamlined conversational UI using Streamlit

## Key Features
Intent classification using Logistic Regression and TF-IDF vectorization

- NLP-based preprocessing with spaCy

- Predefined responses mapped to classified intents

- Follow-up question generation using noun phrase extraction

- Web-based chatbot interface using Streamlit

## Constraints
- No usage of LLMs or external APIs (e.g., OpenAI, Gemini, etc.)

- Must function offline with a local model and dataset

- Initial version is rule-based and uses a static dataset of intents and responses

## Future Scope
- Upgrade to LLM-powered chatbot for dynamic answer generation

- Add memory for long conversation context

- Integrate backend APIs to fetch live job listings, project portfolios, etc.

