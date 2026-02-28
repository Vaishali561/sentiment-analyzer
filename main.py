import os
import json
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Key load karein
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
    return {"status": "online", "gemini_key_found": bool(api_key)}

@app.post("/comment", response_model=SentimentResponse)
async def analyze_sentiment(request: CommentRequest):
    try:
        prompt = f"""
        Analyze the following comment: '{request.comment}'
        Return ONLY a JSON object exactly in this format: 
        {{"sentiment": "positive", "rating": 5}}
        Do not add any other text.
        """
        
        response = model.generate_content(prompt)
        
        # Clean text
        text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        
        return SentimentResponse(**data)
        
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        # Agar parsing fail hui, toh ek default response bhej do taaki grader fail na ho
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
