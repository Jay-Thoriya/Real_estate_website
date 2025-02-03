import streamlit as st
import openai
from brain import get_index_for_pdf  # Assuming this is a custom module
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import asyncio
from openai import AsyncOpenAI

# Load environment variables from the .env file
load_dotenv()

# Set up the OpenAI API key from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')
api_key = os.getenv('OPENAI_API_KEY')

# Check if the API key is set
if not api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set")

# Initialize the AsyncOpenAI client with the API key
client = AsyncOpenAI(api_key=api_key)

st.title("RAG enhanced Chatbot")

@st.cache_resource
def create_vectordb(files, filenames):
    # Show a spinner while creating the vectordb
    with st.spinner("Vector database"):
        vectordb = get_index_for_pdf(
            [file.getvalue() for file in files], filenames, openai.api_key
        )
    return vectordb

pdf_files = st.file_uploader("", type="pdf", accept_multiple_files=True)

if pdf_files:
    pdf_file_names = [file.name for file in pdf_files]
    st.session_state["vectordb"] = create_vectordb(pdf_files, pdf_file_names)

prompt_template = """
    - You are a smart property advisor, guiding users to find the most suitable property from your oldTable and mumbaiTable 
- Engage users by asking targeted questions to understand their needs (e.g., budget, location, property type, BHK)
- Ask one question at a time to quickly filter properties based on user preferences
- After each question, query the inventory to fetch relevant properties, and present the top options that match the user's criteria
- You don't know anything about phones other than the products in the oldTable and mumbaiTable.
- When presenting properties, use a clear, concise sales pitch with key details (name, location, BHK, photo, etc.)
- Provide 2–3 properties that best match their preferences at the end ( below property show only if is exist , if not exist then don't show : 
property_name, location, proprty_by,	min_price_range, bhk, property_type,	carpet_area_in_square_feet,	nearby_location,	total_project_area_in_acres,	no_of_floors,	about_residency,	amenities,	project_specification,	min_cost	max_cost
)
"""

prompt = st.session_state.get("prompt", [{"role": "system", "content": "none"}])

for message in prompt:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

question = st.chat_input("Ask anything")

async def main():
    if question:
        vectordb = st.session_state.get("vectordb", None)
        if not vectordb:
            with st.message("assistant"):
                st.write("You need to provide a PDF")
                return

        search_results = vectordb.similarity_search(question, k=3)
        pdf_extract = "\n".join([result.page_content for result in search_results])

        prompt[0] = {
            "role": "system",
            "content": prompt_template.format(pdf_extract=pdf_extract),
        }

        prompt.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            botmsg = st.empty()

        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=prompt,
            stream=False  
        )

        result = response.choices[0].message.content
        botmsg.write(result)

        prompt.append({"role": "assistant", "content": result})

        st.session_state["prompt"] = prompt

# Run the main function
asyncio.run(main())
