# DeskResearcher
Automate your academic research with AI, Qdrant and arXiv.

## Features
- Automatic arXiv search
- Semantic search with Qdrant
- AI synthesis with Groq (Llama 3.1)
- Email report delivery
- 100% free APIs

## Local Setup

# 1. Clone repository
git clone <your-repo>
cd deskresearcher

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 5. Start Qdrant
docker-compose up -d

# 6. Run application
uvicorn app.main:app --reload
```

## Required API Keys

- **Groq**: https://console.groq.com/keys (free)
- **Resend**: https://resend.com/api-keys (free, 100 emails/day)
- **Qdrant**

## Architecture
```
User → Frontend → FastAPI → Research Engine → Qdrant
                          ↓
                       Groq API
                          ↓
                    Email Service
License
MIT