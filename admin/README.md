# Admin Dashboard

This is the admin dashboard for managing the AI Chatbot application, built with React, TypeScript, and Material-UI.

## Project Structure

```
admin/
├── public/                 # Static files
└── src/
    ├── components/         # Reusable UI components
    ├── features/           # Feature modules
    │   ├── dashboard/     # Dashboard overview
    │   ├── users/         # User management
    │   ├── models/        # AI model management
    │   ├── settings/      # System settings
    │   └── logs/          # System logs
    ├── layouts/           # Layout components
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

2. Create a `.env` file in the admin directory:
   ```
   VITE_API_URL=http://localhost:8000/api/v1
   VITE_ADMIN_PATH=/admin
   ```

3. Start the development server:
   ```bash
   npm run dev
   # or
   yarn dev
   # or
   pnpm dev
   ```

4. Open [http://localhost:3001/admin](http://localhost:3001/admin) in your browser.

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
- [Material-UI](https://mui.com/) - UI components
- [React Query](https://tanstack.com/query) - Data fetching
- [Redux Toolkit](https://redux-toolkit.js.org/) - State management
- [React Router](https://reactrouter.com/) - Routing
- [Axios](https://axios-http.com/) - HTTP client
- [React Hook Form](https://react-hook-form.com/) - Form handling
- [Zod](https://zod.dev/) - Schema validation
- [Recharts](https://recharts.org/) - Data visualization
