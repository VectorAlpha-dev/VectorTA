import React, { useState, useEffect, useRef, useMemo } from 'react';
import { createPortal } from 'react-dom';
import Fuse from 'fuse.js';
import { CategoryIcon } from '../ui/CategoryIcon';

interface SearchItem {
  id: string;
  title: string;
  description: string;
  category: string;
  url: string;
  type: 'indicator' | 'guide' | 'page';
}

interface SearchModalProps {
  isOpen: boolean;
  onClose: () => void;
  searchItems: SearchItem[];
}

export function SearchModal({ isOpen, onClose, searchItems }: SearchModalProps) {
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  // Initialize Fuse.js for fuzzy search
  const fuse = useMemo(() => {
    return new Fuse(searchItems, {
      keys: [
        { name: 'title', weight: 0.4 },
        { name: 'description', weight: 0.3 },
        { name: 'category', weight: 0.2 },
        { name: 'id', weight: 0.1 }
      ],
      threshold: 0.3,
      includeScore: true,
      minMatchCharLength: 2
    });
  }, [searchItems]);

  // Search results
  const results = useMemo(() => {
    if (!query.trim()) {
      // Show recent/popular items when no query
      return searchItems.slice(0, 8).map(item => ({ item, score: 0 }));
    }
    
    return fuse.search(query).slice(0, 20);
  }, [query, fuse, searchItems]);

  // Handle keyboard navigation
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          setSelectedIndex(prev => Math.min(prev + 1, results.length - 1));
          break;
        case 'ArrowUp':
          e.preventDefault();
          setSelectedIndex(prev => Math.max(prev - 1, 0));
          break;
        case 'Enter':
          e.preventDefault();
          if (results[selectedIndex]) {
            window.location.href = results[selectedIndex].item.url;
          }
          break;
        case 'Escape':
          e.preventDefault();
          onClose();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, results, selectedIndex, onClose]);

  // Focus input when modal opens
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
      setQuery('');
      setSelectedIndex(0);
    }
  }, [isOpen]);

  // Scroll selected item into view
  useEffect(() => {
    if (resultsRef.current && results.length > 0) {
      const selectedElement = resultsRef.current.children[selectedIndex] as HTMLElement;
      if (selectedElement) {
        selectedElement.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      }
    }
  }, [selectedIndex, results.length]);

  if (!isOpen) return null;

  const modalContent = (
    <div 
      className="fixed inset-0 z-50 overflow-y-auto"
      onClick={onClose}
    >
      <div className="min-h-screen px-4 text-center">
        {/* Backdrop */}
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm transition-opacity" />

        {/* Center modal */}
        <span className="inline-block h-screen align-middle" aria-hidden="true">
          &#8203;
        </span>

        <div
          className="inline-block w-full max-w-2xl my-8 text-left align-middle transition-all transform bg-background rounded-xl shadow-2xl border border-border"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Search Input */}
          <div className="flex items-center px-4 py-3 border-b border-border">
            <svg 
              className="w-5 h-5 text-muted-foreground mr-3 flex-shrink-0" 
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
            
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search indicators, documentation, and guides..."
              className="flex-1 bg-transparent outline-none text-foreground placeholder-muted-foreground"
            />
            
            <kbd className="hidden sm:inline-flex items-center gap-1 px-2 py-1 text-xs bg-muted rounded">
              <span className="text-xs">ESC</span>
            </kbd>
          </div>

          {/* Search Results */}
          <div 
            ref={resultsRef}
            className="max-h-[60vh] overflow-y-auto"
          >
            {results.length > 0 ? (
              results.map((result, index) => {
                const { item } = result;
                return (
                  <a
                    key={item.id}
                    href={item.url}
                    className={`block px-4 py-3 hover:bg-muted/50 transition-colors ${
                      index === selectedIndex ? 'bg-muted' : ''
                    }`}
                    onMouseEnter={() => setSelectedIndex(index)}
                  >
                    <div className="flex items-start gap-3">
                      <div className="flex-shrink-0 mt-0.5">
                        {item.type === 'indicator' ? (
                          <CategoryIcon category={item.category} className="w-5 h-5 text-primary" />
                        ) : (
                          <svg className="w-5 h-5 text-muted-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" 
                            />
                          </svg>
                        )}
                      </div>
                      
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-foreground">
                            {item.title}
                          </span>
                          {result.score !== undefined && result.score < 0.2 && (
                            <span className="text-xs text-primary">Best match</span>
                          )}
                        </div>
                        <p className="text-sm text-muted-foreground line-clamp-1">
                          {item.description}
                        </p>
                        <div className="flex items-center gap-2 mt-1">
                          <span className="text-xs text-muted-foreground capitalize">
                            {item.type === 'indicator' ? item.category.replace(/_/g, ' ') : item.type}
                          </span>
                        </div>
                      </div>
                      
                      {index === selectedIndex && (
                        <kbd className="hidden sm:inline-flex items-center gap-1 px-2 py-1 text-xs bg-muted rounded">
                          <span className="text-xs">Enter</span>
                        </kbd>
                      )}
                    </div>
                  </a>
                );
              })
            ) : query.trim() ? (
              <div className="px-4 py-8 text-center text-muted-foreground">
                <p>No results found for "{query}"</p>
                <p className="text-sm mt-2">Try searching for indicator names or categories</p>
              </div>
            ) : null}
          </div>

          {/* Footer */}
          <div className="px-4 py-3 border-t border-border">
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <div className="flex items-center gap-4">
                <span className="flex items-center gap-1">
                  <kbd className="px-1.5 py-0.5 bg-muted rounded">↑</kbd>
                  <kbd className="px-1.5 py-0.5 bg-muted rounded">↓</kbd>
                  Navigate
                </span>
                <span className="flex items-center gap-1">
                  <kbd className="px-1.5 py-0.5 bg-muted rounded">Enter</kbd>
                  Open
                </span>
              </div>
              <span>{results.length} results</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  // Use portal to render modal at document root
  return createPortal(modalContent, document.body);
}