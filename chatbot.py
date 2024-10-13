import gradio as gr
from model_RAG import get_rag_response

# Define the chatbot response function
def chatbot_response(inputs , history):
    return get_rag_response(inputs)

# Set up the Gradio UI
with gr.Blocks(fill_height=True) as demo:
    chatbot = gr.ChatInterface(
        fn=chatbot_response,
        title="🍿 Movie Recommendation Assistant 🎥",
    )

if __name__ == "__main__":
    demo.launch(share=True)