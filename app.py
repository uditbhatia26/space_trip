import streamlit as st
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')



# Initialize LLM model
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
openai_llm = ChatOpenAI(model='gpt-4o', api_key=openai_api_key)

embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')

session_id = "Default Session"

if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "Assistant", "content":"Greetings, explorer! Chart your course among the stars, and I shall guide you through the cosmos..."}]

if 'store' not in st.session_state:
    st.session_state.store = {}

# Set page configuration  
st.set_page_config(page_title='Space Expedition Planner', page_icon="✨")

# Title and Subheader
st.title('🚀 Space Expedition Planner')
st.subheader('Plan Your Interplanetary Mission with AI')

# Description
st.markdown(
    """
    **Welcome to the AI-powered Space Mission Planner!**  
    🌍 Select your destination, payload weight, and budget.  
    🛰️ Let AI optimize fuel, trajectory, and mission success.  
    💡 Get a detailed itinerary for your interplanetary journey!  
    """
)

# User Inputs
planet = st.selectbox("🌌 Choose Your Destination", ["Mars", "Europa", "Titan", "Enceladus", "Moon"])
payload_weight = st.slider("📦 Payload Weight (kg)", min_value=100, max_value=50000, step=100)
budget = st.number_input("💰 Budget (in million USD)", min_value=10, max_value=10000, step=10)

   

loader = TextLoader('dataset.txt')
content = loader.load()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=200)
text = splitter.split_documents(content)
vectordb = FAISS.from_documents(text, embedding=embeddings)
retriever = vectordb.as_retriever()

def get_session_history(session_id):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()

    return st.session_state.store[session_id]

prompt_template = """
Given a chat history and the latest user query, rewrite the query into a standalone question that is fully self-contained and does not rely on prior context. Do not answer the question—only rephrase or return it as is if no changes are needed.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", prompt_template),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])


history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt)


sys_prompt = (
    "You are an advanced AI assistant specializing in space exploration and interplanetary missions."
    "You provide scientifically accurate responses while optimizing mission parameters."
    "Consider fuel efficiency, planetary alignments, mission safety, and cost-effectiveness in your answers."
    "Use the provided context to enhance your responses, and if you lack information, respond with: 'I do not have enough data to answer that precisely.'"
    "\n\n"
    "{context}"
)


q_a_prompt = ChatPromptTemplate(
    [
        ("system", sys_prompt),
        MessagesPlaceholder("chat_history"),
        ('user', '{input}'),
    ]
)


question_ans_chain = create_stuff_documents_chain(llm, q_a_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_ans_chain)


conversational_chain = RunnableWithMessageHistory(
    rag_chain, get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer',
)

for message in st.session_state.messages:
        st.chat_message(message['role']).write(message['content'])

if user_input := st.chat_input(placeholder="Ask about your space mission details..."):
    st.chat_message('user').write(user_input)
    st.session_state.messages.append({'role':'user', 'content':user_input})
    response = conversational_chain.invoke(
        {'input': user_input},
        config={'configurable': {'session_id': session_id}}
    )
    with st.chat_message('assistant'):
        st.session_state.messages.append({'role': 'assistant', 'content':response['answer']})
        st.write(response['answer'])
