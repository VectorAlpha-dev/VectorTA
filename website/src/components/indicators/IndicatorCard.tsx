import type { IndicatorCardProps } from '../../types/indicator';

export function IndicatorCard({ indicator, showStatus = false, compact = false }: IndicatorCardProps) {
  const statusColors = {
    'complete': 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
    'in-progress': 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
    'planned': 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200'
  };

  const categoryIcons = {
    'trend': '📈',
    'momentum': '⚡',
    'oscillators': '〰️',
    'moving_averages': '📊',
    'volatility': '📉',
    'volume': '📊',
    'other': '🔧'
  };

  return (
    <a
      href={`/indicators/${indicator.id}`}
      className={`
        group block p-6 bg-card rounded-lg shadow-sm border border-border
        hover:shadow-lg hover:border-primary/50 transition-all duration-200
        ${compact ? 'p-4' : 'p-6'}
      `}
      aria-label={`View documentation for ${indicator.title}`}
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-2xl" aria-hidden="true">
              {categoryIcons[indicator.category]}
            </span>
            <h3 className="font-semibold text-lg group-hover:text-primary transition-colors truncate">
              {indicator.title}
            </h3>
          </div>
          
          {!compact && (
            <p className="text-sm text-muted-foreground mb-3 line-clamp-2">
              {indicator.description}
            </p>
          )}
          
          <div className="flex items-center gap-4 text-xs text-muted-foreground">
            <span className="font-mono">{indicator.id}</span>
            <span className="flex items-center gap-1">
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                  d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
              </svg>
              {indicator.parameters.length} params
            </span>
            {showStatus && (
              <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${statusColors[indicator.implementationStatus]}`}>
                {indicator.implementationStatus}
              </span>
            )}
          </div>
        </div>
        
        <svg 
          className="w-5 h-5 text-muted-foreground group-hover:text-primary group-hover:translate-x-1 transition-all flex-shrink-0"
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24"
          aria-hidden="true"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
      </div>
    </a>
  );
}