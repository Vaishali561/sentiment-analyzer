import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
from typing import Literal
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    return {"status": "online", "key_configured": bool(api_key)}

@app.post("/comment", response_model=SentimentResponse)
async def analyze_sentiment(request: CommentRequest):
    if not api_key:
        raise HTTPException(status_code=500, detail="API Key not configured")
        
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a sentiment analyzer. Return JSON with sentiment (positive, negative, neutral) and rating (1-5)."},
                {"role": "user", "content": request.comment},
            ],
            response_format=SentimentResponse,
        )
        return completion.choices[0].message.parsed
        
    except Exception as e:
        # ASLI ERROR YAHAN LOG HOGA
        error_msg = traceback.format_exc()
        print(f"CRITICAL ERROR:\n{error_msg}")
        # Crash hone ke bajaye error detail bhejo taaki grader bataye kya hua
        raise HTTPException(status_code=500, detail=f"OpenAI Error: {str(e)}")
