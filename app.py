import os
import pinecone
import openai
from flask import Flask, request, jsonify, render_template, send_from_directory
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from flask_cors import CORS
from waitress import serve
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
#from make_it import docs

os.environ["OPENAI_API_KEY"] = ""
application = Flask(__name__)
CORS(application)

pinecone.init(api_key="",environment="")  # Initialize Pinecone
index_name = ''  # Specify the Pinecone index name
#pinecone_index = pinecone.Index(index_name=index_name)  # Create the Pinecone index
embeddings = OpenAIEmbeddings(openai_api_key="")
index = Pinecone.from_existing_index(index_name=index_name,embedding=embeddings)

# Create an OpenAI instance
openai_instance = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# @application.route('/')
# def index():
#     return render_template('index.html')

@application.route('/process_question', methods=['POST'])
def process_question():
    # Get the question from the request
    data = request.get_json()
    question = data ['question']
    def get_similiar_docs(query, k=10, score=False):
        if score:
            similar_docs = index.similarity_search_with_score(question, k=k)
        else:
            similar_docs = index.similarity_search(query, k=k)
        return similar_docs
    model_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(openai_api_key="",model_name=model_name)

    chain = load_qa_chain(llm, chain_type="stuff")
    similar_docs = get_similiar_docs(question)
    answer = chain.run(input_documents=similar_docs, question=question)
        
    
    # response = openai.Embedding.create(input=question, model='text-embedding-ada-002')
    # embedding = response['data'][0]['embedding']
    #print(embedding)
    # Load the question answering chain
    # chain = load_qa_chain(openai_instance, chain_type="stuff")

    # # Query Pinecone for relevant chunks
    #results = pinecone_index.query(vector=[embedding], top_k=5,namespace='')
    # print(results)
    # # Get the embeddings of the retrieved chunks
    # embeddings = [result['values'] for result in results['matches']]
    # answer=embeddings
    # Perform the question answering
    #answer = chain.run(input_documents=results.ids, question=question, embeddings=embeddings)

    # Return the answer as a JSON response
    return jsonify({'answer': answer})

@application.route('/.well-known/ai-plugin.json')
def serve_ai_plugin():
    return send_from_directory('.well-known', 'ai-plugin.json', mimetype='application/json')

@application.route('/.well-known/openapi.yaml')
def serve_openapi_yaml():
    return send_from_directory('.well-known', 'openapi.yaml', mimetype='text/yaml')

@application.route('/.well-known/logo.png')
def serve_logo_png():
    return send_from_directory('.well-known', 'logo.png')

if __name__ == '__main__':
    serve(application, host="0.0.0.0", port=5000)
