import streamlit as st
from langchain.llms import OpenAI
from langchain import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv
import os

load_dotenv()

st.title("YouTube Video Script Generator")

key = st.text_input('Plug in your OpenAI API key (Optional)')
prompt = st.text_input("Enter the prompt here")

if key:
    os.environ['OPENAI_API_KEY'] = key
    llm = OpenAI(temperature = 0.6)
else:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0, "max_length":512})
# Prompt Template

title_template = PromptTemplate(
    input_variables=['topic'],
    template="Write me a YouTube Video Title about {topic}"
)

script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'],
    template = 'Write me a YouTube Script based on the Title {title} in 1000 words. while leveraging this Wikipedia Research {wikipedia_research}'
)

title_memory = ConversationBufferMemory(input_key = 'topic',
                                        memory_key = 'chat_history')

script_memory = ConversationBufferMemory(input_key = 'title',
                                        memory_key = 'chat_history')


title_chain = LLMChain(llm = llm, 
                       prompt = title_template,
                       output_key='title',
                       memory = title_memory,
                       verbose = True)

script_chain = LLMChain(llm=llm,
                        prompt = script_template,
                        output_key='script',
                        memory = script_memory,
                        verbose = True)

wiki = WikipediaAPIWrapper()

if prompt:
    data_load_state = st.text('Loading...')
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title = title, 
                              wikipedia_research = wiki_research)
    st.write(title)
    st.write(script)
    data_load_state = st.text('Title and Script Generated Successfully...')