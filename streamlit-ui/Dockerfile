FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# Using ${PORT} ensures compatibility with Google Cloud's auto-assigned port
ENTRYPOINT ["streamlit", "run", "StreamlitApp.py", "--server.port=${PORT}", "--server.address=0.0.0.0"]
