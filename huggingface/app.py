import os
import nest_asyncio
import pandas as pd
import frontmatter
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from llama_index.llms.openai import OpenAI
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    Document,
    VectorStoreIndex,
    get_response_synthesizer,
    PromptTemplate,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import gradio as gr

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Pinecone and OpenAI clients
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
client = OpenAI(model="gpt-4o-mini", temperature=0)

# Global configuration for LLM and embedding models
embedding = OpenAIEmbedding(model="text-embedding-ada-002")
Settings.llm = client
Settings.embed_model = embedding
Settings.chunk_size_limit = 1536

# Create the vector store using Pinecone
pinecone_index = pinecone_client.Index("chatbot-index")
vector_store = PineconeVectorStore(pinecone_index)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
retriever = VectorIndexRetriever(index=index, similarity_top_k=5)

# Define the chatbot's prompt template
prompt_template = (
    "You are a friendly chatbot specialized in helping beginners use FamilySearch and its tools. 😊 "
    "This includes record hints, source attachments, and other related topics. Provide clear and concise answers, "
    "and try to make the conversation enjoyable! 😄\n\n"
    "Context:\n"
    "#####################################\n"
    "{context_str}\n"
    "Answer the user's question: {query_str}\n\n"
    "If the question is related to FamilySearch or its tools (such as record hints, source attachments, or genealogical research), "
    "provide a detailed answer along with a summary. Also, include the following source metadata as 'Source':\n"
    "- **Title**: {title}\n"
    "- **Publish Date**: {date}\n"
    "- **URL**: {url}\n\n"
    "However, if the question is unrelated to FamilySearch, provide a direct and concise answer without any summary or metadata."
)

# Create the template and response synthesizer
qa_template = PromptTemplate(template=prompt_template)
response_synthesizer = get_response_synthesizer(
    llm=client, text_qa_template=qa_template, response_mode="compact"
)
query_engine = RetrieverQueryEngine(
    retriever=retriever, response_synthesizer=response_synthesizer
)


# Define a function to handle chatbot responses
def respond(message, history):
    try:
        # Query the engine and extract the response
        response = query_engine.query(message)
        information = (
            response.response
        )  # Ensure to extract the textual content from the response object
    except Exception as e:
        # Handle any errors that occur during the response generation
        information = f"Error processing the request: {str(e)}"

    # Append the user's message and the chatbot's response to the chat history
    history.append((message, information))
    return "", history


# Initial welcome message to introduce the chatbot's function
intro_message = "👋 Hello! I'm here to help you with any questions about using FamilySearch! 🌟 Feel free to ask me anything about navigating the tools like record hints, source attachments, and genealogical research."


# Configure the Gradio interface
with gr.Blocks() as demo:
    # Initialize the chatbot with an introductory message
    chatbot = gr.Chatbot(value=[["", intro_message]])
    btn = gr.Button("Enviar")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    # Link the 'send' button to the respond function
    btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])

    # Allow users to press Enter to submit their message
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(share=True)
