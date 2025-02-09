FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install -r ./requirements.txt

CMD ["streamlit", "run", "chat.py", "--server.port=8501", "--server.address=0.0.0.0"]