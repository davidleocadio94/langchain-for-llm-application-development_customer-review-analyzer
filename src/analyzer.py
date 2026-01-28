"""LangChain Customer Review Analyzer - Chains, Memory, and Structured Output"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain


# Structured output for review analysis
sentiment_schema = ResponseSchema(
    name="sentiment",
    description="positive, negative, or neutral"
)
summary_schema = ResponseSchema(
    name="summary",
    description="One sentence summary of the review"
)
key_issues_schema = ResponseSchema(
    name="key_issues",
    description="List of any problems mentioned (empty list if none)"
)
is_urgent_schema = ResponseSchema(
    name="is_urgent",
    description="true if customer needs immediate help, false otherwise"
)

response_schemas = [sentiment_schema, summary_schema, key_issues_schema, is_urgent_schema]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)


def analyze_review(review_text: str) -> dict:
    """Analyze a customer review and return structured output."""
    llm = ChatOpenAI(temperature=0)

    format_instructions = output_parser.get_format_instructions()

    review_template = """
    For the following customer review, extract the following information:

    sentiment: Is the sentiment positive, negative, or neutral?
    summary: Provide a one sentence summary of the review.
    key_issues: List any problems mentioned (empty list if none).
    is_urgent: Is immediate help needed (true or false)?

    text: {review_text}

    {format_instructions}
    """

    prompt = ChatPromptTemplate.from_template(template=review_template)
    messages = prompt.format_messages(
        review_text=review_text,
        format_instructions=format_instructions
    )

    response = llm(messages)
    output_dict = output_parser.parse(response.content)

    return output_dict


def analyze_and_respond(review_text: str) -> dict:
    """Analyze a review using a sequential chain and generate a response."""
    llm = ChatOpenAI(temperature=0)

    # Chain 1 - Analyze Review
    first_prompt = ChatPromptTemplate.from_template(
        """Analyze the following customer review and provide:
        - Sentiment (positive, negative, or neutral)
        - Main issue (if any)
        - Urgency level (high, medium, or low)

        Review: {review}
        """
    )
    chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key="analysis")

    # Chain 2 - Detect Language
    second_prompt = ChatPromptTemplate.from_template(
        """What language is the following review written in? Just provide the language name.

        Review: {review}
        """
    )
    chain_two = LLMChain(llm=llm, prompt=second_prompt, output_key="language")

    # Chain 3 - Generate Response
    third_prompt = ChatPromptTemplate.from_template(
        """Based on the following analysis and detected language, write a professional customer service response.
        Write the response in {language}.

        Analysis: {analysis}

        Customer service response:
        """
    )
    chain_three = LLMChain(llm=llm, prompt=third_prompt, output_key="response")

    # Create Sequential Chain
    overall_chain = SequentialChain(
        chains=[chain_one, chain_two, chain_three],
        input_variables=["review"],
        output_variables=["analysis", "language", "response"],
        verbose=False
    )

    result = overall_chain({"review": review_text})
    return result


# Conversation memory for interactive analysis
class ReviewAssistant:
    """Interactive review analysis assistant with memory."""

    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        self.memory = ConversationBufferMemory()
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=False
        )

    def chat(self, message: str) -> str:
        """Send a message and get a response."""
        return self.conversation.predict(input=message)

    def clear_memory(self):
        """Clear conversation history."""
        self.memory.clear()


# Global assistant instance
assistant = ReviewAssistant()


def chat_with_assistant(message: str) -> str:
    """Chat with the review assistant."""
    return assistant.chat(message)


def clear_assistant_memory():
    """Clear the assistant's memory."""
    assistant.clear_memory()
    return "Memory cleared."
