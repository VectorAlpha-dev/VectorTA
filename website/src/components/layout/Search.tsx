import { useState, useEffect, useRef } from 'react';
import { indicators } from '../../data/indicator-registry';

export function Search() {
  const [isExpanded, setIsExpanded] = useState(false);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<any[]>([]);
  const [showResults, setShowResults] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setIsExpanded(true);
        setShowResults(true);
      }
      if (e.key === 'Escape') {
        setIsExpanded(false);
        setShowResults(false);
        setQuery('');
      }
    };

    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setShowResults(false);
        if (!query) {
          setIsExpanded(false);
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    document.addEventListener('mousedown', handleClickOutside);
    
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [query]);

  useEffect(() => {
    if (isExpanded && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isExpanded]);

  useEffect(() => {
    if (query) {
      const filtered = Object.entries(indicators)
        .filter(([id, indicator]) => 
          indicator.name.toLowerCase().includes(query.toLowerCase()) ||
          id.includes(query.toLowerCase()) ||
          indicator.category.includes(query.toLowerCase())
        )
        .slice(0, 8)
        .map(([id, indicator]) => ({ id, ...indicator }));
      
      setResults(filtered);
    } else {
      setResults([]);
    }
  }, [query]);

  const handleSearchClick = () => {
    setIsExpanded(true);
    setShowResults(true);
  };

  return (
    <div ref={containerRef} className="relative">
      <div className={`flex items-center transition-all duration-200 ${isExpanded ? 'w-80' : 'w-auto'}`}>
        {!isExpanded ? (
          <button
            onClick={handleSearchClick}
            className="flex items-center gap-2 px-3 py-1.5 text-sm text-muted-foreground bg-muted hover:bg-muted/80 rounded-lg transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <span className="hidden sm:inline">Search</span>
            <kbd className="hidden md:inline-flex items-center gap-1 px-1.5 py-0.5 text-xs bg-background border border-border rounded">
              <span className="text-xs">⌘</span>K
            </kbd>
          </button>
        ) : (
          <div className="flex items-center gap-2 w-full px-3 py-1.5 bg-muted rounded-lg">
            <svg className="w-4 h-4 text-muted-foreground flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onFocus={() => setShowResults(true)}
              placeholder="Search indicators..."
              className="flex-1 bg-transparent outline-none text-sm text-foreground placeholder-muted-foreground"
              autoComplete="off"
            />
            <button
              onClick={() => {
                setIsExpanded(false);
                setShowResults(false);
                setQuery('');
              }}
              className="p-1 hover:bg-muted-foreground/20 rounded transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        )}
      </div>

      {/* Results dropdown */}
      {isExpanded && showResults && (query || results.length > 0) && (
        <div className="absolute top-full mt-2 right-0 w-80 max-h-96 overflow-y-auto bg-card rounded-lg shadow-lg border border-border z-50">
          {results.length > 0 ? (
            results.map((result) => (
              <a
                key={result.id}
                href={`/indicators/${result.id}`}
                className="block px-4 py-3 hover:bg-muted/50 transition-colors border-b border-border last:border-0 focus:bg-muted/50 focus:outline-none"
                onClick={() => {
                  setIsExpanded(false);
                  setShowResults(false);
                  setQuery('');
                }}
              >
                <div className="font-medium text-sm text-foreground">{result.name}</div>
                <div className="text-xs text-muted-foreground">
                  {result.category.replace(/_/g, ' ')} • {result.id}
                </div>
              </a>
            ))
          ) : query ? (
            <div className="p-6 text-center text-muted-foreground">
              <p className="text-sm">No indicators found for "{query}"</p>
            </div>
          ) : (
            <div className="p-6 text-center text-muted-foreground">
              <p className="text-sm">Start typing to search {Object.keys(indicators).length} indicators</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}