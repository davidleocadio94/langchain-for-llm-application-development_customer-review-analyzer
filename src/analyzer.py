"""LangChain Customer Review Analyzer - Chains, Memory, and Structured Output"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import json
import re


def analyze_review(review_text: str) -> dict:
    """Analyze a customer review and return structured output."""
    llm = ChatOpenAI(temperature=0)

    review_template = """
    For the following customer review, extract the following information and return as JSON:

    - sentiment: Is the sentiment positive, negative, or neutral?
    - summary: Provide a one sentence summary of the review.
    - key_issues: List any problems mentioned (empty list if none).
    - is_urgent: Is immediate help needed (true or false as string)?

    Review: {review_text}

    Return ONLY valid JSON in this exact format:
    {{"sentiment": "...", "summary": "...", "key_issues": [...], "is_urgent": "true" or "false"}}
    """

    prompt = ChatPromptTemplate.from_template(template=review_template)
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({"review_text": review_text})

    # Parse JSON from response
    try:
        # Find JSON in response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            output_dict = json.loads(json_match.group())
        else:
            output_dict = json.loads(response)
    except json.JSONDecodeError:
        output_dict = {
            "sentiment": "unknown",
            "summary": response,
            "key_issues": [],
            "is_urgent": "false"
        }

    return output_dict


def analyze_and_respond(review_text: str) -> dict:
    """Analyze a review using chained prompts and generate a response."""
    llm = ChatOpenAI(temperature=0)
    parser = StrOutputParser()

    # Step 1 - Analyze Review
    analysis_prompt = ChatPromptTemplate.from_template(
        """Analyze the following customer review and provide:
        - Sentiment (positive, negative, or neutral)
        - Main issue (if any)
        - Urgency level (high, medium, or low)

        Review: {review}
        """
    )
    analysis_chain = analysis_prompt | llm | parser
    analysis = analysis_chain.invoke({"review": review_text})

    # Step 2 - Detect Language
    language_prompt = ChatPromptTemplate.from_template(
        """What language is the following review written in? Just provide the language name.

        Review: {review}
        """
    )
    language_chain = language_prompt | llm | parser
    language = language_chain.invoke({"review": review_text})

    # Step 3 - Generate Response
    response_prompt = ChatPromptTemplate.from_template(
        """Based on the following analysis and detected language, write a professional customer service response.
        Write the response in {language}.

        Analysis: {analysis}

        Customer service response:
        """
    )
    response_chain = response_prompt | llm | parser
    response = response_chain.invoke({"analysis": analysis, "language": language})

    return {
        "review": review_text,
        "analysis": analysis,
        "language": language,
        "response": response
    }


# Conversation memory for interactive analysis
class ReviewAssistant:
    """Interactive review analysis assistant with memory."""

    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        self.history = ChatMessageHistory()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful customer service assistant that helps analyze and respond to customer reviews."),
            ("placeholder", "{history}"),
            ("human", "{input}")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def chat(self, message: str) -> str:
        """Send a message and get a response."""
        # Get history as list of messages
        history_messages = self.history.messages

        # Invoke chain with history
        response = self.chain.invoke({
            "input": message,
            "history": history_messages
        })

        # Add to history
        self.history.add_user_message(message)
        self.history.add_ai_message(response)

        return response

    def clear_memory(self):
        """Clear conversation history."""
        self.history.clear()


# Global assistant instance
assistant = ReviewAssistant()


def chat_with_assistant(message: str) -> str:
    """Chat with the review assistant."""
    return assistant.chat(message)


def clear_assistant_memory():
    """Clear the assistant's memory."""
    assistant.clear_memory()
    return "Memory cleared."
