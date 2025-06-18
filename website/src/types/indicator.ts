export interface IndicatorParameter {
  name: string;
  type: 'number' | 'boolean' | 'string';
  default: number | boolean | string;
  min?: number;
  max?: number;
  description: string;
}

export interface IndicatorReturn {
  type: string;
  description: string;
}

export interface IndicatorReference {
  title: string;
  url: string;
}

export type IndicatorCategory = 
  | 'trend'
  | 'momentum'
  | 'volatility'
  | 'volume'
  | 'moving_averages'
  | 'oscillators'
  | 'other';

export type ImplementationStatus = 'complete' | 'in-progress' | 'planned';

export interface Indicator {
  id: string;
  title: string;
  description: string;
  category: IndicatorCategory;
  parameters: IndicatorParameter[];
  returns: IndicatorReturn;
  complexity: string;
  implementationStatus: ImplementationStatus;
  references?: IndicatorReference[];
  seeAlso?: string[];
  since?: string;
}

export interface IndicatorCardProps {
  indicator: Indicator;
  showStatus?: boolean;
  compact?: boolean;
}