import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

st.set_page_config(page_title="Text To Math Problem Solver And Data Search Assistant", page_icon="🧮")
st.title("Text To Math Problem Solver Using Google Gemini 🚀")

# Sidebar API key input
gemini_api_key = st.sidebar.text_input(label="Google Gemini API Key", type="password")

if not gemini_api_key:
    st.info("Please add your Gemini API key to continue")
    st.stop()

# LLM initialization
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_api_key, temperature=0)

# ---------------------------
# TOOLS
# ---------------------------

# Wikipedia Search Tool
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool to search Wikipedia for facts and information."
)

# Math Calculator Tool
math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for solving math expressions."
)

# Reasoning Tool
prompt = """
You are an intelligent math reasoning assistant.
Solve the question step-by-step in bullet points.
Show clear logic and the final answer.

Question: {question}
Answer:
"""

prompt_template = PromptTemplate(input_variables=["question"], template=prompt)
chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning Tool",
    func=chain.run,
    description="A tool for detailed reasoning-based mathematical solutions."
)

# ---------------------------
# AGENT
# ---------------------------
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# ---------------------------
# CHAT HISTORY
# ---------------------------

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a Math chatbot powered by Gemini. Ask me any math or knowledge-based question!"}
    ]

# Display history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# ---------------------------
# USER INPUT
# ---------------------------

question = st.text_input("Enter your problem here")

if st.button("Find my answer"):
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.spinner("Generating response..."):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

            # FIXED: Pass ONLY the question, not message history
            response = assistant_agent.run(question, callbacks=[st_cb])

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

    else:
        st.warning("Please enter a question.")

