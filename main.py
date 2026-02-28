import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini API Setup
# API Key aap Google AI Studio (aistudio.google.com) se le sakte hain
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    rating: int = Field(..., ge=1, le=5)

class CommentRequest(BaseModel):
    comment: str

@app.get("/")
async def root():
    return {"status": "online", "model": "gemini-1.5-flash"}

@app.post("/comment", response_model=SentimentResponse)
async def analyze_sentiment(request: CommentRequest):
    try:
        # Instruction tak ki output sirf JSON mile
        prompt = f"""
        Analyze this comment: '{request.comment}'
        Return ONLY a JSON object with:
        sentiment: 'positive', 'negative', or 'neutral'
        rating: 1-5
        """
        
        response = model.generate_content(prompt)
        
        # Gemini ke text response ko dict mein convert karein
        # (Yahan aap manual parsing ya json.loads use kar sakte hain)
        import json
        result = json.loads(response.text.replace("```json", "").replace("```", ""))
        
        return SentimentResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
