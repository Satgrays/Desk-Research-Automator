"""
FastAPI Backend for DeskResearcher
"""
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import os
from dotenv import load_dotenv

from app.research_engine import ResearchEngine
from app.email_service import send_research_report

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="DeskResearcher API",
    description="Automate desk research with AI and Qdrant",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Initialize research engine (singleton)
try:
    research_engine = ResearchEngine()
    print("Research Engine initialized correctly")
except Exception as e:
    print(f"Error initializing Research Engine: {e}")
    research_engine = None


# Pydantic models
class ResearchRequest(BaseModel):
    query: str
    email: EmailStr
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Recent advances in quantum computing for cryptography",
                "email": "researcher@example.com"
            }
        }


class ResearchResponse(BaseModel):
    status: str
    message: str


# Background task for asynchronous processing
def process_research_task(query: str, email: str):
    """
    Background task that executes the complete research
    """
    try:
        # Execute research
        result = research_engine.run_full_research(query)
        
        if result['status'] == 'success':
            # Send email with results
            email_sent = send_research_report(
                to_email=email,
                query=query,
                report=result['report'],
                sources=result['sources']
            )
            
            if email_sent:
                print(f"Complete process successful for {email}")
            else:
                print(f"Research completed but email failed for {email}")
        else:
            print(f"Research failed: {result.get('message')}")
            
    except Exception as e:
        print(f"Error in background task: {e}")


# Endpoints
@app.get("/", response_class=FileResponse)
async def root():
    """Serve the frontend"""
    return FileResponse("app/static/index.html")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    qdrant_status = "connected" if research_engine else "disconnected"
    return {
        "status": "ok",
        "qdrant": qdrant_status,
        "groq": "configured" if os.getenv("GROQ_API_KEY") else "not configured",
        "resend": "configured" if os.getenv("RESEND_API_KEY") else "not configured"
    }


@app.post("/api/research", response_model=ResearchResponse)
async def create_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """
    Start a new research
    
    The process executes in background and results are sent by email
    """
    # Validate that engine is initialized
    if not research_engine:
        raise HTTPException(
            status_code=503,
            detail="Research engine not available. Check Qdrant configuration."
        )
    
    # Validate API keys
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY not configured"
        )
    
    if not os.getenv("RESEND_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="RESEND_API_KEY not configured"
        )
    
    # Validate query
    if len(request.query.strip()) < 10:
        raise HTTPException(
            status_code=400,
            detail="Query must be at least 10 characters"
        )
    
    # Add background task
    background_tasks.add_task(process_research_task, request.query, request.email)
    
    return ResearchResponse(
        status="processing",
        message=f"Your research is being processed. You will receive an email at {request.email} in 1-3 minutes."
    )


@app.get("/api/status")
async def get_status():
    """System information"""
    return {
        "app": "DeskResearcher",
        "version": "1.0.0",
        "components": {
            "research_engine": "active" if research_engine else "inactive",
            "qdrant": "connected" if research_engine else "disconnected",
            "groq_api": "configured" if os.getenv("GROQ_API_KEY") else "missing",
            "resend_api": "configured" if os.getenv("RESEND_API_KEY") else "missing"
        }
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)