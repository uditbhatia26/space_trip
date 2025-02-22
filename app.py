import streamlit as st
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize LLM models
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
openai_llm = ChatOpenAI(model='gpt-4o', api_key=openai_api_key)

# Load embeddings and create retriever
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
loader = TextLoader('dataset.txt')
content = loader.load()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=200)
text = splitter.split_documents(content)
vectordb = FAISS.from_documents(text, embedding=embeddings)
retriever = vectordb.as_retriever()

# Session-based history tracking
session_id = "Default Session"
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'itinerary_generated' not in st.session_state:
    st.session_state.itinerary_generated = False

if 'store' not in st.session_state:
    st.session_state.store = {}

# Streamlit page config
st.set_page_config(page_title='Space Expedition Planner', page_icon="✨")

# UI Elements
st.title('🚀 Space Expedition Planner')
st.subheader('Plan Your Interplanetary Mission with AI')

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

# Function to retrieve session history
def get_session_history(session_id):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# Prompt for history-aware retriever
history_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user query into a standalone question."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

history_aware_retriever = create_history_aware_retriever(llm, retriever, history_prompt)

# System prompt for answering questions
sys_prompt = (
    "You are an advanced AI assistant specializing in space exploration and interplanetary missions."
    "You provide scientifically accurate responses while optimizing mission parameters."
    "Consider fuel efficiency, planetary alignments, mission safety, and cost-effectiveness."
    "Use the provided context to enhance your responses."
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

# Function to generate itinerary
def generate_itinerary(planet, payload_weight, budget):
    prompt = f"""
    You are an AI space mission planner. Generate an itinerary for the following mission:

    - Destination: {planet}
    - Payload Weight: {payload_weight} kg
    - Budget: ${budget} million

    Include:
    - 🚀 Launch details
    - 🛰️ Spacecraft requirements
    - ⏳ Mission duration
    - 🔥 Fuel considerations
    - ⚠️ Expected challenges
    - 🌍 Landing & return strategy
    - 🔬 Scientific goals

    Keep the response engaging yet concise.
    """
    return llm.invoke(prompt).content

# Generate itinerary once all options are selected
if not st.session_state.itinerary_generated and st.button("Generate Itinerary 🚀"):
    with st.spinner("Calculating your space expedition..."):
        itinerary = generate_itinerary(planet, payload_weight, budget)
        st.session_state.messages.append({"role": "assistant", "content": itinerary})
        st.session_state.itinerary_generated = True

# Display chat history
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

# Chatbot interaction after itinerary is generated
if st.session_state.itinerary_generated:
    if user_input := st.chat_input(placeholder="Ask about your space mission details..."):
        st.chat_message("user").write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        response = conversational_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )

        with st.chat_message("assistant"):
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})
            st.write(response['answer'])
