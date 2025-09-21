import React from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from './button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from './dropdown-menu';
import { Globe } from 'lucide-react';

const languages = [
  { code: 'en', name: 'English' },
  { code: 'es', name: 'Español' },
];

export function LanguageSelector() {
  const { i18n } = useTranslation();
  const currentLanguage = languages.find(lang => lang.code === i18n.language) || languages[0];

  const changeLanguage = (language) => {
    i18n.changeLanguage(language);
    // Save language preference to localStorage
    localStorage.setItem('userLanguage', language);
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon" className="h-9 w-9">
          <Globe className="h-4 w-4" />
          <span className="sr-only">Change language</span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        {languages.map((lang) => (
          <DropdownMenuItem
            key={lang.code}
            onClick={() => changeLanguage(lang.code)}
            className={currentLanguage.code === lang.code ? 'bg-accent' : ''}
          >
            {lang.name}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
