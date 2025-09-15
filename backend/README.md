# AI Chatbot Backend

This is the backend service for the AI Chatbot application, built with FastAPI.

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py        # Application configuration
│   │   └── security.py      # Authentication and authorization
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py          # Dependencies
│   │   └── api_v1/          # API v1 endpoints
│   ├── models/
│   │   ├── __init__.py
│   │   └── user.py          # User model
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── user.py          # Pydantic models
│   └── services/
│       ├── __init__.py
│       └── user_service.py  # Business logic
├── tests/                   # Test files
├── requirements.txt         # Python dependencies
└── .env.example            # Example environment variables
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. Run the development server:
   ```bash
   uvicorn app.main:app --reload
   ```

## Development

- API documentation will be available at `/docs`
- Run tests with `pytest`
- Format code with `black .`
- Check for linting issues with `flake8`
