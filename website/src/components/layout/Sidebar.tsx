import React, { useState, useEffect } from 'react';
import { CategoryIcon } from '../ui/CategoryIcon';
import { indicators } from '../../data/indicator-registry';

interface SidebarProps {
  currentPath?: string;
}

export function Sidebar({ currentPath = '' }: SidebarProps) {
  const [searchQuery, setSearchQuery] = useState('');

  // Get all indicators sorted alphabetically
  const allIndicators = React.useMemo(() => {
    return Object.entries(indicators)
      .map(([id, indicator]) => ({ id, name: indicator.name }))
      .sort((a, b) => a.name.localeCompare(b.name));
  }, []);

  // Filter indicators based on search
  const filteredIndicators = React.useMemo(() => {
    if (!searchQuery) return allIndicators;
    
    const query = searchQuery.toLowerCase();
    return allIndicators.filter(ind => 
      ind.name.toLowerCase().includes(query) ||
      ind.id.toLowerCase().includes(query)
    );
  }, [allIndicators, searchQuery]);

  return (
    <aside className="w-64 h-full bg-background border-r border-border overflow-hidden flex flex-col">
      {/* Search */}
      <div className="p-4 border-b border-border">
        <div className="relative">
          <input
            type="text"
            placeholder="Filter indicators..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full px-3 py-2 pl-9 text-sm bg-muted rounded-lg outline-none focus:ring-2 focus:ring-primary/20"
          />
          <svg 
            className="absolute left-3 top-2.5 w-4 h-4 text-muted-foreground" 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" 
            />
          </svg>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto py-4">
        {/* Header */}
        <div className="px-4 mb-6">
          <h2 className="text-lg font-semibold">Indicators</h2>
          <p className="text-sm text-muted-foreground mt-1">
            {Object.keys(indicators).length} total
          </p>
        </div>

        {/* Indicators List */}
        <div className="px-4">
          {filteredIndicators.map((indicator) => {
            const isActive = currentPath === `/indicators/${indicator.id}`;
            return (
              <a
                key={indicator.id}
                href={`/indicators/${indicator.id}`}
                className={`block px-3 py-1.5 rounded-md text-sm transition-colors ${
                  isActive
                    ? 'bg-primary/10 text-primary font-medium'
                    : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                }`}
              >
                {indicator.name}
              </a>
            );
          })}
        </div>
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-border">
        <a 
          href="/guides"
          className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
              d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" 
            />
          </svg>
          <span>Documentation</span>
        </a>
      </div>
    </aside>
  );
}

