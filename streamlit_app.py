import streamlit as st
import requests

# API endpoint for the local LLM chat service
API_URL = "http://localhost:8000/chat"

# Streamlit app title
st.title("Local LLM Cocktail Advisor")

# Chat history in session state
if "history" not in st.session_state:
    st.session_state["history"] = []

# User input field
msg = st.text_input("Your message:")

# Handle message submission
if st.button("Send"):
    if msg.strip():
        try:
            resp = requests.post(API_URL, json={"user_message": msg.strip()})
            if resp.status_code == 200:
                data = resp.json()
                answer = data["answer"]
                sources = data.get("sources", [])
                st.session_state["history"].append((msg, answer, sources))
            else:
                st.error(f"Error: {resp.status_code} - Unable to fetch response from API.")

        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

for idx, (q, a, src) in enumerate(st.session_state["history"]):
    st.write(f"**User:** {q}")
    st.write(f"**Assistant:** {a}")

    if src:
        st.write(f"**Sources:** {', '.join(src)}")
    
    st.write("---")
