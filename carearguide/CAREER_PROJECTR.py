# --- SQLite patch for Chroma on Streamlit Cloud ---
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
# ---------------------------------------------------
#import necessary libraries
import google.generativeai as genai
import streamlit as st

# ✅ Page title and header

st.title("🎓 AI Career Path Advisor")
st.write("Fill in your preferences and get a personalized career roadmap!")

#API Configuration

genai.configure(api_key="AIzaSyBE-lpNACAu_b6RV5UMLbm2PGXj73wqniw")
model = genai.GenerativeModel('gemini-1.5-flash')

def career_advice(tech, career, course, timeperiod):
    prompt = f"""
    your are my personal AI assistant that helps me with career advice.
    I'm planning to start a career in {tech}
    My goal is to become a {career}
    I want to pursue a course in {course}
    I want to achieve this goal in {timeperiod}
    
    Please Suggest me 
    1. The best learning path to achieve my goal
    2. The best resources to learn from
    3. The best projects to work on
    4. The best way to build a portfolio
    5. The best way to network and find job opportunities
    6. The best way to prepare for interviews
    7. Any other tips or advice to help me succeed in my career
    8. Suggest me a weekly schedule to follow to achieve my goal in the given time
    """
    response = model.generate_content(prompt)
    return response.text if response else "Sorry, I couldn't process that request."

# # Main execution
# if _name_ == "_main_":
#     tech = input("Enter the Technology you want to start a career in: ")
#     career = input("Enter the career you want to pursue: ")
#     course = input("Enter the course you want to pursue: ")
#     timeperiod = input("Enter the time period you want to achieve your goal in: ")

#     advice = career_advice(tech, career, course, timeperiod)
#     print(advice)

# Streamlit UI
tech = st.selectbox("Select Technology", ["AI", "Data Science", "Web Development", "Cybersecurity", "Cloud Computing"])
career = st.text_input("Enter Career Goal (e.g. Data Scientist, ML Engineer, Web Developer)")
course = st.selectbox("Select Course Type", ["Bootcamp", "Certification", "Degree", "Self-paced Learning"])
timeperiod = st.selectbox("Select Time Period", ["3 months", "6 months", "1 year", "2 years"])

if st.button("Generate Career Plan 🚀"):
    if career.strip() == "":
        st.warning("⚠ Please enter a career goal before generating the plan.")
    else:
        with st.spinner("Generating your personalized roadmap..."):
            advice = career_advice(tech, career, course, timeperiod)
            st.success("✅ Here’s your Career Roadmap:")
            st.write(advice)





