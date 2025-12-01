import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, Tool
from langchain_community.utilities import WikipediaAPIWrapper
import numexpr as ne

st.set_page_config(page_title="Math + Knowledge Assistant", page_icon="🧮")
st.title("Text → Math Solver & Knowledge Assistant 🚀")

# Sidebar API Key
gemini_api_key = st.sidebar.text_input("Google Gemini API Key", type="password")

if not gemini_api_key:
    st.info("Please enter your Gemini API key to continue.")
    st.stop()

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_api_key,
    temperature=0
)

# -----------------------
# 1️⃣ Wikipedia Search Tool
# -----------------------
wiki = WikipediaAPIWrapper()
wiki_tool = Tool(
    name="Wikipedia",
    func=wiki.run,
    description="Search Wikipedia for facts and information."
)

# -----------------------
# 2️⃣ Math Calculator Tool (WORKING)
# -----------------------
def safe_calculator(expression: str):
    """Evaluate math expressions safely using numexpr."""
    try:
        result = ne.evaluate(expression).item()
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

calculator_tool = Tool(
    name="Calculator",
    func=safe_calculator,
    description="Solve mathematical expressions. Example: 2+3*4"
)

# -----------------------
# 3️⃣ Reasoning Tool
# -----------------------
prompt = """
You are a math reasoning assistant.
Solve step-by-step using bullet points and give the final answer.

Question: {question}
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

reason_chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning Tool",
    func=reason_chain.run,
    description="Detailed reasoning for math solutions."
)

# -----------------------
# AGENT
# -----------------------
assistant_agent = initialize_agent(
    tools=[wiki_tool, calculator_tool, reasoning_tool],
    llm=llm,
    agent="zero-shot-react-description",
    handle_parsing_errors=True,
    verbose=False
)

# -----------------------
# Chat UI
# -----------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I can solve math & search info for you. Ask anything!"}
    ]

for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

question = st.text_input("Enter your problem:")

if st.button("Solve"):
    if not question:
        st.warning("Please enter a question.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.spinner("Thinking..."):
            response = assistant_agent.run(question)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
