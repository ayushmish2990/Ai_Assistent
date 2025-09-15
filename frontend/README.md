# AI Chatbot Frontend

This is the user-facing frontend for the AI Chatbot application, built with React and TypeScript.

## Project Structure

```
frontend/
├── public/                 # Static files
└── src/
    ├── assets/             # Static assets (images, fonts, etc.)
    ├── components/         # Reusable UI components
    ├── features/           # Feature-based modules
    │   ├── auth/          # Authentication
    │   ├── chat/          # Chat interface
    │   ├── models/        # AI model management
    │   └── settings/      # User settings
    ├── lib/               # Utility functions
    ├── pages/             # Page components
    ├── services/          # API services
    ├── store/             # State management
    ├── types/             # TypeScript type definitions
    ├── App.tsx            # Main App component
    └── main.tsx           # Application entry point
```

## Setup

1. Install dependencies:
   ```bash
   npm install
   # or
   yarn
   # or
   pnpm install
   ```

2. Create a `.env` file in the frontend directory:
   ```
   VITE_API_URL=http://localhost:8000/api/v1
   VITE_WS_URL=ws://localhost:8000/ws
   ```

3. Start the development server:
   ```bash
   npm run dev
   # or
   yarn dev
   # or
   pnpm dev
   ```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Available Scripts

- `dev` - Start the development server
- `build` - Build for production
- `preview` - Preview the production build
- `test` - Run tests
- `lint` - Run ESLint
- `format` - Format code with Prettier

## Tech Stack

- [React](https://reactjs.org/) - UI library
- [TypeScript](https://www.typescriptlang.org/) - Type checking
- [Vite](https://vitejs.dev/) - Build tool
- [Tailwind CSS](https://tailwindcss.com/) - Styling
- [React Query](https://tanstack.com/query) - Data fetching
- [Zustand](https://github.com/pmndrs/zustand) - State management
- [React Router](https://reactrouter.com/) - Routing
- [Axios](https://axios-http.com/) - HTTP client
- [React Hook Form](https://react-hook-form.com/) - Form handling
- [Zod](https://zod.dev/) - Schema validation
