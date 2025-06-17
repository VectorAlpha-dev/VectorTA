import { useState, useEffect, useRef } from 'react';
import { indicators } from '../../data/indicator-registry';

export function Search() {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<any[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setIsOpen(true);
      }
      if (e.key === 'Escape') {
        setIsOpen(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  useEffect(() => {
    if (query) {
      const filtered = Object.entries(indicators)
        .filter(([id, indicator]) => 
          indicator.name.toLowerCase().includes(query.toLowerCase()) ||
          id.includes(query.toLowerCase()) ||
          indicator.category.includes(query.toLowerCase())
        )
        .slice(0, 10)
        .map(([id, indicator]) => ({ id, ...indicator }));
      
      setResults(filtered);
    } else {
      setResults([]);
    }
  }, [query]);

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className="flex items-center space-x-2 px-3 py-2 text-sm text-gray-600 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
        <span>Search</span>
        <kbd className="hidden md:inline-block px-2 py-1 text-xs bg-gray-200 dark:bg-gray-700 rounded">⌘K</kbd>
      </button>

      {isOpen && (
        <div className="fixed inset-0 z-50 bg-black/50" onClick={() => setIsOpen(false)}>
          <div 
            className="mx-auto mt-20 max-w-2xl bg-white dark:bg-gray-800 rounded-lg shadow-2xl"
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center p-4 border-b dark:border-gray-700">
              <svg className="w-5 h-5 text-gray-400 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <input
                ref={inputRef}
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search indicators..."
                className="flex-1 bg-transparent outline-none"
              />
              <kbd className="px-2 py-1 text-xs bg-gray-100 dark:bg-gray-700 rounded">ESC</kbd>
            </div>

            {results.length > 0 && (
              <div className="max-h-96 overflow-y-auto">
                {results.map(result => (
                  <a
                    key={result.id}
                    href={`/indicators/${result.id}`}
                    className="block px-4 py-3 hover:bg-gray-100 dark:hover:bg-gray-700 border-b dark:border-gray-700"
                  >
                    <div className="font-medium">{result.name}</div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                      {result.category.replace(/_/g, ' ')}
                    </div>
                  </a>
                ))}
              </div>
            )}

            {query && results.length === 0 && (
              <div className="p-8 text-center text-gray-500">
                No indicators found for "{query}"
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}