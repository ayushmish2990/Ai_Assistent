import { NavLink } from 'react-router-dom';
import { cn } from '@/lib/utils';

export function NavItem({ to, icon, children }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        cn(
          'flex items-center px-4 py-2 text-sm font-medium rounded-md transition-colors',
          isActive
            ? 'bg-accent text-accent-foreground'
            : 'text-foreground/70 hover:bg-accent/50 hover:text-foreground'
        )
      }
    >
      {icon && <span className="mr-3">{icon}</span>}
      {children}
    </NavLink>
  );
}
