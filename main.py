from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
import os

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. Define the Structured Schema
class SentimentResponse(BaseModel):
    sentiment: str = Field(..., pattern="^(positive|negative|neutral)$")
    rating: int = Field(..., ge=1, le=5)

class CommentRequest(BaseModel):
    comment: str

@app.post("/comment", response_model=SentimentResponse)
async def analyze_sentiment(request: CommentRequest):
    try:
        # 2. Call OpenAI with Structured Output enforcement
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Analyze the sentiment and rating of the provided customer comment."},
                {"role": "user", "content": request.comment},
            ],
            response_format=SentimentResponse,
        )

        # 3. Access the parsed object directly
        return completion.choices[0].message.parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))