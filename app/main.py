from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes import generate
from app.services.providers import FalProvider, Veo3Provider
from app.services.providers.openrouter_provider import OpenRouterProvider
from app.services.pipeline_orchestrator import init_pipeline_orchestrator

app = FastAPI(
    title="AI Generation API",
    description="Backend service for AI content generation",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ugcvideo-app.netlify.app",  # Netlify production frontend
        "https://ugcvideo-app-client-production.up.railway.app",  # Railway frontend (backup)
        "https://ugcvideo-app-server-production.up.railway.app",  # Production backend
        "http://localhost:3000",  # Local development
        "http://localhost:3001",  # Local development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize providers and pipeline orchestrator on startup"""
    print("\n" + "="*60)
    print("🚀 Initializing AI Generation API")
    print("="*60)

    # Initialize providers with API keys from settings
    print("📦 Initializing Fal.ai provider (Stage 1 & 2 - Seedream models)...")
    fal_provider = FalProvider()

    print("📦 Initializing Veo3 provider (Stage 3 with fallback)...")
    veo3_provider = Veo3Provider(
        kie_api_key=settings.KIE_API_KEY,
        fal_api_key=settings.FAL_KEY
    )

    # Initialize OpenRouter provider (optional, for GPT-4o prompt enhancement)
    openrouter_provider = None
    if hasattr(settings, 'OPENROUTER_API_KEY') and settings.OPENROUTER_API_KEY != "your-openrouter-api-key-here":
        try:
            print("📦 Initializing OpenRouter provider (GPT-4o prompt enhancement)...")
            openrouter_provider = OpenRouterProvider(
                api_key=settings.OPENROUTER_API_KEY,
                model=settings.OPENROUTER_MODEL
            )
            print("✅ OpenRouter provider initialized - GPT-4o prompt enhancement ENABLED")
        except Exception as e:
            print(f"⚠️  Warning: Failed to initialize OpenRouter provider: {e}")
            print("⚠️  Continuing without GPT-4o prompt enhancement...")
    else:
        print("⚠️  OpenRouter API key not configured - GPT-4o prompt enhancement DISABLED")

    # Initialize pipeline orchestrator
    print("🔧 Initializing pipeline orchestrator...")
    init_pipeline_orchestrator(
        fal_provider=fal_provider,
        veo3_provider=veo3_provider,
        openrouter_provider=openrouter_provider
    )

    print("✅ All providers initialized successfully!")
    print("="*60 + "\n")


# Auth dependency
async def verify_api_key(authorization: str = Header(...)):
    """Verify API key from Authorization header"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header format")

    token = authorization.replace("Bearer ", "")
    if token != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return True


# Include routes with authentication
app.include_router(
    generate.router,
    prefix="/api/v1",
    tags=["generation"],
    dependencies=[Depends(verify_api_key)]
)


@app.get("/health")
async def health_check():
    """Health check endpoint (no auth required)"""
    return {
        "status": "healthy",
        "service": "ai-generation-api",
        "version": "1.0.0"
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Generation API",
        "docs": "/docs",
        "health": "/health"
    }
