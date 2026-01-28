"""Gradio interface for Customer Review Analyzer."""

import gradio as gr
from src.analyzer import analyze_review, analyze_and_respond, chat_with_assistant, clear_assistant_memory


with gr.Blocks(title="Customer Review Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Customer Review Analyzer

        Analyze customer reviews using LangChain with structured output, sequential chains, and conversation memory.

        **Three modes available:**
        1. **Quick Analysis** - Extract sentiment, summary, issues, and urgency
        2. **Full Pipeline** - Analyze, detect language, and generate response
        3. **Interactive Chat** - Conversational analysis with memory
        """
    )

    with gr.Tabs():
        # Tab 1: Quick Analysis
        with gr.TabItem("Quick Analysis"):
            gr.Markdown("### Extract structured information from a review")
            review_input1 = gr.Textbox(
                label="Customer Review",
                placeholder="Enter a customer review to analyze...",
                lines=4
            )
            analyze_btn1 = gr.Button("Analyze Review", variant="primary")

            with gr.Row():
                sentiment_out = gr.Textbox(label="Sentiment")
                urgent_out = gr.Textbox(label="Urgent?")

            summary_out = gr.Textbox(label="Summary")
            issues_out = gr.Textbox(label="Key Issues")

            def quick_analyze(review):
                if not review.strip():
                    return "", "", "", ""
                result = analyze_review(review)
                return (
                    result.get("sentiment", ""),
                    str(result.get("is_urgent", "")),
                    result.get("summary", ""),
                    str(result.get("key_issues", []))
                )

            analyze_btn1.click(
                fn=quick_analyze,
                inputs=review_input1,
                outputs=[sentiment_out, urgent_out, summary_out, issues_out]
            )

        # Tab 2: Full Pipeline
        with gr.TabItem("Full Pipeline"):
            gr.Markdown("### Analyze review + detect language + generate response")
            review_input2 = gr.Textbox(
                label="Customer Review",
                placeholder="Enter a customer review (any language)...",
                lines=4
            )
            analyze_btn2 = gr.Button("Run Full Pipeline", variant="primary")

            analysis_out = gr.Textbox(label="Analysis", lines=3)
            language_out = gr.Textbox(label="Detected Language")
            response_out = gr.Textbox(label="Generated Response", lines=5)

            def full_pipeline(review):
                if not review.strip():
                    return "", "", ""
                result = analyze_and_respond(review)
                return (
                    result.get("analysis", ""),
                    result.get("language", ""),
                    result.get("response", "")
                )

            analyze_btn2.click(
                fn=full_pipeline,
                inputs=review_input2,
                outputs=[analysis_out, language_out, response_out]
            )

        # Tab 3: Interactive Chat
        with gr.TabItem("Interactive Chat"):
            gr.Markdown("### Chat with the review assistant (with memory)")
            chatbot = gr.Chatbot(height=400)
            msg = gr.Textbox(
                label="Your message",
                placeholder="Share a review or ask questions about it...",
                lines=2
            )
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear Memory", variant="secondary")

            def respond(message, chat_history):
                if not message.strip():
                    return "", chat_history
                response = chat_with_assistant(message)
                chat_history.append((message, response))
                return "", chat_history

            def clear_chat():
                clear_assistant_memory()
                return []

            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            send_btn.click(respond, [msg, chatbot], [msg, chatbot])
            clear_btn.click(clear_chat, outputs=chatbot)

    # Example reviews
    gr.Markdown("### Example Reviews")
    gr.Examples(
        examples=[
            ["This product is terrible! It broke after one day and customer service was useless. I want my money back!"],
            ["J'adore ce produit! La qualité est exceptionnelle et la livraison était très rapide."],
            ["The software works okay but the documentation is confusing. It took me hours to set up."],
            ["¡Excelente servicio! El equipo de soporte fue muy amable y resolvió mi problema rápidamente."],
        ],
        inputs=review_input1,
    )

if __name__ == "__main__":
    demo.launch()
