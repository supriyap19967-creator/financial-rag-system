import streamlit as st
import requests

API_URL = "https://rag-api-912628415543.us-central1.run.app"

st.title("Financial Regulation RAG Assistant")

question = st.text_input("Ask a question about financial regulations:")

if st.button("Submit"):

    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={"question": question},
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()
                st.success(data.get("answer", "No answer returned"))

            else:
                st.error(f"API Error: {response.status_code}")
                st.text(response.text)

        except Exception as e:
            st.error("⚠ Backend API not reachable.")
            st.text(str(e))
