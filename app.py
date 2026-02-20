from fastapi import FastAPI,Path,Query
from pydantic import BaseModel, Field
from typing import Annotated
from rag1 import ask_question



app=FastAPI()
@app.get("/")
def home():
    return {"message": "YouTube RAG API is running"}
class Question(BaseModel):
    url:Annotated[str,Field(...,description="give the video id from the link")]
    que: Annotated[str,Field(...,description="Ask the question regarding the video link")]

@app.get("/health")
def health():
    return {"status":"running"}

# question1=question()

@app.post("/ask-que")
def ask_que(data:Question):
    answer=ask_question(data.url,data.que)

    return {"answer":answer}