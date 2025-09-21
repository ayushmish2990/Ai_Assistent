import { Routes, Route, Navigate } from 'react-router-dom';
import { lazy, Suspense } from 'react';
import { Loader2 } from 'lucide-react';

// Lazy load admin components
const AdminDashboard = lazy(() => import('@/pages/admin/dashboard'));
const UserManagement = lazy(() => import('@/pages/admin/users'));
const SystemSettings = lazy(() => import('@/pages/admin/settings'));

const LoadingFallback = () => (
  <div className="flex items-center justify-center h-64">
    <Loader2 className="h-8 w-8 animate-spin text-primary" />
  </div>
);

export default function AdminRoutes() {
  return (
    <Suspense fallback={<LoadingFallback />}>
      <Routes>
        <Route index element={<AdminDashboard />} />
        <Route path="users" element={<UserManagement />} />
        <Route path="settings" element={<SystemSettings />} />
        <Route path="*" element={<Navigate to="/admin" replace />} />
      </Routes>
    </Suspense>
  );
}
