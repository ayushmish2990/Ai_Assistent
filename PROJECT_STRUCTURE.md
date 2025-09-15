# AI Coding Assistant - Project Structure

This document outlines the structure of the AI Coding Assistant project, which consists of multiple components working together to provide a comprehensive AI-powered development environment.

## Repository Structure

```
ai-coding-assistant/
├── backend/                  # Python FastAPI backend
│   ├── app/
│   │   ├── api/             # API routes and endpoints
│   │   ├── core/            # Core functionality and configuration
│   │   ├── crud/            # Database operations
│   │   ├── db/              # Database models and session
│   │   ├── models/          # SQLAlchemy models
│   │   ├── schemas/         # Pydantic models
│   │   └── services/        # Business logic
│   ├── tests/               # Backend tests
│   ├── .env.example         # Example environment variables
│   ├── requirements.txt     # Python dependencies
│   └── README.md            # Backend documentation
│
├── frontend/                # User-facing React application
│   ├── public/              # Static files
│   └── src/
│       ├── assets/          # Images, fonts, etc.
│       ├── components/      # Reusable UI components
│       ├── features/        # Feature modules
│       ├── lib/             # Utility functions
│       ├── pages/           # Page components
│       ├── services/        # API services
│       ├── store/           # State management
│       ├── types/           # TypeScript types
│       ├── App.tsx          # Main App component
│       └── main.tsx         # Application entry point
│
├── admin/                   # Admin dashboard
│   ├── public/              # Static files
│   └── src/
│       ├── components/      # Reusable UI components
│       ├── features/        # Admin feature modules
│       ├── layouts/         # Layout components
│       ├── lib/             # Utility functions
│       ├── pages/           # Page components
│       ├── services/        # API services
│       ├── store/           # State management
│       ├── types/           # TypeScript types
│       ├── App.tsx          # Main App component
│       └── main.tsx         # Application entry point
│
├── docs/                    # Project documentation
├── scripts/                 # Utility scripts
└── README.md                # Main project documentation
```

## Component Descriptions

### Backend (FastAPI)
- **API**: RESTful endpoints for the frontend and admin interfaces
- **Core**: Application configuration, security, and utilities
- **CRUD**: Database operations and business logic
- **Models**: SQLAlchemy ORM models
- **Schemas**: Pydantic models for request/response validation
- **Services**: Business logic and external service integrations

### Frontend (React + TypeScript)
- **Features**: Self-contained feature modules
- **Components**: Reusable UI components
- **Pages**: Top-level page components
- **Services**: API client and service layer
- **Store**: Global state management
- **Types**: TypeScript type definitions

### Admin Dashboard (React + TypeScript)
- **Features**: Admin-specific functionality
- **Layouts**: Page layouts and navigation
- **Components**: Admin UI components
- **Pages**: Admin page components
- **Services**: Admin API services
- **Store**: Admin state management

## Development Setup

### Prerequisites
- Python 3.9+
- Node.js 16+
- PostgreSQL 13+
- npm/yarn/pnpm

### Getting Started

1. **Backend Setup**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env
   # Edit .env with your configuration
   uvicorn app.main:app --reload
   ```

2. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   cp .env.example .env
   # Edit .env with your API URL
   npm run dev
   ```

3. **Admin Setup**
   ```bash
   cd admin
   npm install
   cp .env.example .env
   # Edit .env with your API URL
   npm run dev
   ```

## Deployment

### Backend
- Deploy using Docker or a WSGI server like Gunicorn with Uvicorn workers
- Set up a PostgreSQL database
- Configure environment variables

### Frontend & Admin
- Build for production: `npm run build`
- Deploy the `dist` folder to a static file server or CDN

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## License

MIT
