# Blackbox AI Admin Dashboard

A modern, responsive admin dashboard for managing the Blackbox AI application. Built with React, Tailwind CSS, and Radix UI.

## Features

- 📊 Dashboard with system statistics and activity overview
- 👥 User management with search, filtering, and pagination
- ⚙️ System settings with multiple configuration tabs
- 🔒 Role-based access control
- 🌓 Dark/light theme support
- 📱 Fully responsive design

## Tech Stack

- ⚛️ React 18
- 🎨 Tailwind CSS with dark mode
- 🛣️ React Router v6
- 🎭 Radix UI for accessible components
- 📊 React Query for data fetching
- 📝 React Hook Form for form handling
- 📜 Zod for schema validation
- ✨ Lucide Icons

## Getting Started

### Prerequisites

- Node.js 16 or later
- npm or yarn

### Installation

1. Clone the repository
2. Install dependencies:

```bash
npm install
# or
yarn
```

### Development

Start the development server:

```bash
npm run dev
# or
yarn dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Building for Production

```bash
npm run build
# or
yarn build
```

## Project Structure

```
src/
├── components/         # Reusable UI components
│   ├── layout/        # Layout components
│   └── ui/            # Base UI components
├── contexts/          # React contexts
├── hooks/             # Custom React hooks
├── lib/               # Utility functions
├── pages/             # Page components
│   ├── dashboard.jsx  # Dashboard page
│   ├── users.jsx      # User management page
│   ├── settings.jsx   # Settings page
│   └── not-found.jsx  # 404 page
├── App.jsx            # Main application component
└── main.jsx           # Application entry point
```

## Available Scripts

- `npm run dev` - Start the development server
- `npm run build` - Build for production
- `npm run preview` - Preview the production build
- `npm run lint` - Run ESLint
- `npm run format` - Format code with Prettier

## License

MIT
