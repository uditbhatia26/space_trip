import streamlit as st
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# LLM Instance
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)



# Streamlit UI
st.set_page_config(page_title='Space Expedition Planner', page_icon="✨")

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

planet = st.selectbox("🌌 Choose Your Destination", ["Mars", "Europa", "Titan", "Enceladus", "Moon"])
payload_weight = st.slider("📦 Payload Weight (kg)", min_value=100, max_value=50000, step=100)
budget = st.number_input("💰 Budget (in million USD)", min_value=10, max_value=10000, step=10)


if st.button("Generate Itinerary 🚀"):
    with st.spinner("Calculating your space expedition..."):
        prompt = f"""
        You are an advanced AI assistant specialized in space mission planning.
        A user wants to plan an interplanetary trip with the following parameters:

        - **Destination**: {planet}
        - **Payload Weight**: {payload_weight} kg
        - **Budget**: ${budget} million

        Generate a detailed space itinerary including:
        - 🚀 Launch details
        - 🛰️ Spacecraft requirements
        - ⏳ Mission duration
        - 🔥 Fuel considerations
        - ⚠️ Expected challenges
        - 🌍 Landing & return strategy
        - 🔬 Scientific goals or experiments

        Keep the response concise but informative.
        """
        
        response = llm.invoke(prompt) 

    # Display the generated itinerary
    st.subheader("🛰️ Your Space Expedition Itinerary")
    st.write(response.content)
