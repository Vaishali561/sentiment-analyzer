import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
from typing import Literal

app = FastAPI()

# CORS fix: Ye line request ko allow karti hai
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    rating: int = Field(..., ge=1, le=5)

class CommentRequest(BaseModel):
    comment: str

@app.get("/")
async def root():
    return {"status": "online"}

@app.post("/comment")
async def analyze_sentiment(request: CommentRequest):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Analyze sentiment (positive, negative, neutral) and rating (1-5)."},
            {"role": "user", "content": request.comment},
        ],
        response_format=SentimentResponse,
    )
    return completion.choices[0].message.parsed
