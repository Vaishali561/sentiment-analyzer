import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from typing import Literal

# Initialize FastAPI
app = FastAPI()

# Initialize OpenAI Client
# This looks for the OPENAI_API_KEY you set in the Render Dashboard
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. Define the Schema for Structured Outputs
class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    rating: int = Field(..., ge=1, le=5)

class CommentRequest(BaseModel):
    comment: str

# 2. Add a Home Route (Fixes the 404 on the main URL)
@app.get("/")
async def home():
    return {"status": "online", "message": "Visit /docs for the API interface"}

# 3. The Sentiment Analysis Endpoint
@app.post("/comment", response_model=SentimentResponse)
async def analyze_sentiment(request: CommentRequest):
    if not request.comment:
        raise HTTPException(status_code=400, detail="Comment cannot be empty")
    
    try:
        # Using Beta Parse for guaranteed JSON structure
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Analyze customer feedback. Return sentiment and a rating from 1-5."},
                {"role": "user", "content": request.comment},
            ],
            response_format=SentimentResponse,
        )

        return completion.choices[0].message.parsed

    except Exception as e:
        # This will show up in your Render Logs if the API key is wrong
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="AI Analysis Failed")
