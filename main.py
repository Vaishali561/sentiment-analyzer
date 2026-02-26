import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
from typing import Literal

app = FastAPI()

# --- STEP 1: ADD CORS PERMISSIONS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all websites to access your API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    rating: int = Field(..., ge=1, le=5)

class CommentRequest(BaseModel):
    comment: str

@app.get("/")
async def root():
    return {"status": "online"}

@app.post("/comment", response_model=SentimentResponse)
async def analyze_sentiment(request: CommentRequest):
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Analyze sentiment and rating."},
                {"role": "user", "content": request.comment},
            ],
            response_format=SentimentResponse,
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        # This print will show up in your Render Logs
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))
