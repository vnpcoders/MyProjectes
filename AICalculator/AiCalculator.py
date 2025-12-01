import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Wikipedia Tool
from langchain_community.utilities import WikipediaAPIWrapper

# Calculator Tool (NEW for LangChain 0.3+)
from langchain.tools import PythonREPLTool

# Agent
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType

# Streamlit callback handler
from langchain.callbacks import StreamlitCallbackHandler

from langchain.chains import LLMChain


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Text To Math Problem Solver And Data Search Assistant",
    page_icon="🧮"
)
st.title("Text To Math Problem Solver Using Google Gemini 🚀")


# ---------------------------------------------------------
# API KEY (SIDEBAR)
# ---------------------------------------------------------
gemini_api_key = st.sidebar.text_input(
    label="Google Gemini API Key",
    type="password"
)

if not gemini_api_key:
    st.info("Please add your Gemini API key to continue")
    st.stop()


# ---------------------------------------------------------
# LLM INITIALIZATION
# ---------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_api_key,
    temperature=0
)


# ---------------------------------------------------------
# TOOLS
# ---------------------------------------------------------

# Wikipedia tool
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="Search Wikipedia for facts, history, and knowledge."
)

# Math tool (Python REPL)
calculator = Tool(
    name="Calculator",
    func=PythonREPLTool().run,
    description="Executes Python code for math calculations."
)

# Reasoning tool
prompt = """
You are a detailed reasoning assistant.
Solve the question step-by-step in bullet points.
Show clear logic and then give the final answer.

Question: {question}
Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

reasoning_chain = LLMChain(
    llm=llm,
    prompt=prompt_template
)

reasoning_tool = Tool(
    name="Reasoning Tool",
    func=reasoning_chain.run,
    description="Provides step-by-step reasoning for math problems."
)


# ---------------------------------------------------------
# AGENT INITIALIZATION
# ---------------------------------------------------------
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)


# ---------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm a Gemini-powered math chatbot. Ask me anything!"}
    ]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])


# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------
question = st.text_input("Enter your problem here")

if st.button("Find my answer"):
    if not question:
        st.warning("Please enter a question.")
    else:
        st.session_state["messages"].append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.spinner("Generating response..."):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agent.run(question, callbacks=[st_cb])

        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

