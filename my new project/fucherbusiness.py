import google.generativeai as genai
import streamlit as st

# ✅ Page title and header

st.title("🎓 AI Fucher Business Path Advisor")
st.write("Fill in your preferences and get a Business roadmap!")

#API Configuration
genai.configure(api_key="Enter your API key")
model = genai.GenerativeModel('gemini-2.5-flash')


def career_advice(filled, amount, profit,timeperiod):
    prompt = f"""
    your are my personal AI assistant that helps me with Bast Business advice.
    I'm planning to start a Business in {filled}
    I am able to invest in starting of a business {amount}
    I want to get profit in this percentas compere to seal in month {profit}
    I want to achieve the best growth,stability and profit in  {timeperiod}
    
    Please Suggest me 
    1. The best learning path to 
    2. The best resources to learn business
    3. The best place to start business 
    4. The best way to build a stable business
    5. The best way to connect and find clients 
    6. The best way to prepare for interviews
    7. Any other tips or advice to help me succeed in my life
    8. Suggest me a weekly schedule to follow to achieve my goal in the given time
    9. Also tell me the risk in the business 
    """
    response = model.generate_content(prompt)
    return response.text if response else "Sorry, I couldn't process that request."

# Streamlit UI
filled = st.selectbox("Select Filled ", ["Farming Related", "IT Related", "Marketing Related", "food Related", " textile Related"])
amount = st.text_input("Enter Amount to invest (e.g. 1 lack, 5 lack , 15 lack)")
profit = st.selectbox("Select minimum profit per month ", ["10%", "20%", "30%", "50%"])
timeperiod = st.selectbox("Select Time Period", ["3 months", "6 months", "1 year", "2 years"])

if st.button("Generate Business Plan 🚀"):
    if amount.strip() == "":
        st.warning("⚠ Please enter a investment amount before generating the plan.")
    else:
        with st.spinner("Generating your personalized roadmap..."):
            advice = career_advice(filled, amount, profit, timeperiod)
            st.success("✅ Here’s your Business Roadmap:")
            st.write(advice)

