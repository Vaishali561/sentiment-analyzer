import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from typing import Literal

# Initialize the App
app = FastAPI(title="Sentiment API")

# Initialize OpenAI Client (Uses the Environment Variable you set in Render)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. Define the Schema for Structured Outputs
class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    rating: int = Field(..., ge=1, le=5)

class CommentRequest(BaseModel):
    comment: str

# 2. Fixes the "Not Found" error for the base URL
@app.get("/")
async def home():
    return {
        "status": "online",
        "message": "Send a POST request to /comment",
        "docs": "/docs"
    }

# 3. The Analysis Endpoint
@app.post("/comment", response_model=SentimentResponse)
async def analyze_sentiment(request: CommentRequest):
    if not request.comment.strip():
        raise HTTPException(status_code=400, detail="Comment is empty")
    
    try:
        # Structured Output call
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Analyze customer feedback sentiment and rating (1-5)."},
                {"role": "user", "content": request.comment},
            ],
            response_format=SentimentResponse,
        )
        return completion.choices[0].message.parsed

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="API Analysis Failed")
        
