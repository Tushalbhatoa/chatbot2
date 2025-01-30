from flask import Flask, request, jsonify
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env
API_KEY = os.getenv("HUGGING_FACE_API_KEY")

app = Flask(__name__)
vectorstore = Chroma(persist_directory= "C:\project-II\chatbot2.0\8a511ed8-8a39-4085-b82e-311e5cb0938d")

llm = HuggingFaceHub(repo_id = "deepset/roberta-base-squad2", temperature = 0.7, huggingfacehub_api_token = API_KEY)
chain = ConversationalRetrievalChain(llm = llm, retriever = vectorstore.as_retriever())

@app.route('/chat', methods = ['POST'])
def chat():
    user_message = request.json.get("message")
    response = chain.run(user_message)
    return jsonify({"response": response})
