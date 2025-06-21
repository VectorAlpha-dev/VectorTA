import React, { useState, useEffect } from 'react';
import { SearchModal } from './SearchModal';
import type { SearchItem } from './search-types';

interface SearchButtonProps {
  searchItems: SearchItem[];
}

export function SearchButton({ searchItems }: SearchButtonProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isMac, setIsMac] = useState(false);

  useEffect(() => {
    // Detect if user is on Mac
    setIsMac(navigator.platform.toLowerCase().includes('mac'));

    // Global keyboard shortcut
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setIsOpen(true);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className="group flex items-center gap-2 px-3 py-1.5 text-sm bg-muted hover:bg-muted/70 rounded-lg transition-all"
        aria-label="Search documentation"
      >
        <svg 
          className="w-4 h-4 text-muted-foreground group-hover:text-foreground transition-colors" 
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24"
        >
          <path 
            strokeLinecap="round" 
            strokeLinejoin="round" 
            strokeWidth={2} 
            d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" 
          />
        </svg>
        
        <span className="hidden sm:inline text-muted-foreground group-hover:text-foreground transition-colors">
          Search...
        </span>
        
        <kbd className="hidden sm:inline-flex items-center gap-0.5 px-1.5 py-0.5 text-xs bg-background border border-border rounded">
          {isMac ? '⌘' : 'Ctrl'}K
        </kbd>
      </button>

      <SearchModal 
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        searchItems={searchItems}
      />
    </>
  );
}