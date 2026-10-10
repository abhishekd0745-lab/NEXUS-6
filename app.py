import gradio as gr
import joblib

# Load the trained NEXUS-6 model
model = joblib.load("nexus6_model.joblib")

def detect_news(text):
    if not text or not text.strip():
        return "⚠️ Please enter a news headline or article.", ""

    try:
        prediction = model.predict([text])[0]

        if prediction == 1:
            result = "🚨 Potentially Fake News"
        else:
            result = "✅ Likely Reliable News"

        confidence = ""

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([text])[0]
            confidence = f"Model confidence: {max(probabilities) * 100:.2f}%"

        return result, confidence

    except Exception as e:
        return "Unable to analyze this text.", str(e)


with gr.Blocks(theme=gr.themes.Soft(), title="NEXUS-6 V4") as demo:

    gr.Markdown(
        """
        # ⚡ NEXUS-6 V4
        ### AI-Powered Fake News & Clickbait Detector

        Analyze news headlines and text using our trained machine-learning model.
        """
    )

    news_input = gr.Textbox(
        label="📰 Enter News Headline or Article",
        placeholder="Paste your news headline or article here...",
        lines=6
    )

    analyze_button = gr.Button("🔍 Analyze News", variant="primary")

    result_output = gr.Textbox(label="Analysis Result", interactive=False)
    confidence_output = gr.Textbox(label="Model Confidence", interactive=False)

    analyze_button.click(
        fn=detect_news,
        inputs=news_input,
        outputs=[result_output, confidence_output]
    )

    gr.Markdown(
        """
        ---
        ⚠️ **Note:** This tool provides predictions from a machine-learning model.
        Results are not guaranteed to be accurate. Always verify important news
        with trusted sources.
        """
    )

if __name__ == "__main__":
    demo.launch()
