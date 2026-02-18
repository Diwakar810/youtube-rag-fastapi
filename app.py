from fastapi import FastAPI,Path,Query
from pydantic import BaseModel, Field
from typing import Annotated
from rag import ask_question

app=FastAPI()
class Question(BaseModel):
    url:Annotated[str,Field(...,description="give the video id from the link")]
    que: Annotated[str,Field(...,description="Ask the question regarding the video link")]



# question1=question()

@app.post("/ask-que")
def ask_que(data:Question):
    answer=ask_question(data.url,data.que)

    return {"answer":answer}