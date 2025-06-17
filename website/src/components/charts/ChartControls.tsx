import { useState } from 'react';

interface Parameter {
  name: string;
  type: 'number' | 'boolean';
  default: any;
  min?: number;
  max?: number;
  description?: string;
}

interface ChartControlsProps {
  indicatorId: string;
  parameters: Record<string, any>;
  parameterDefs: Parameter[];
  onParameterChange: (params: Record<string, any>) => void;
}

export function ChartControls({ indicatorId, parameters, parameterDefs, onParameterChange }: ChartControlsProps) {
  const [localParams, setLocalParams] = useState(parameters);

  const handleChange = (paramName: string, value: any) => {
    const newParams = { ...localParams, [paramName]: value };
    setLocalParams(newParams);
    onParameterChange(newParams);
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 space-y-4">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
        Parameters
      </h3>
      
      {parameterDefs.map((param) => (
        <div key={param.name} className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
            {param.name.charAt(0).toUpperCase() + param.name.slice(1).replace(/([A-Z])/g, ' $1').replace(/_/g, ' ')}
            {param.description && (
              <span className="block text-xs text-gray-500 dark:text-gray-400 mt-1">
                {param.description}
              </span>
            )}
          </label>
          
          {param.type === 'number' && (
            <div className="flex items-center space-x-4">
              <input
                type="range"
                min={param.min || 1}
                max={param.max || 100}
                value={localParams[param.name] || param.default}
                onChange={(e) => handleChange(param.name, Number(e.target.value))}
                className="flex-1"
              />
              <input
                type="number"
                min={param.min || 1}
                max={param.max || 100}
                value={localParams[param.name] || param.default}
                onChange={(e) => handleChange(param.name, Number(e.target.value))}
                className="w-20 px-2 py-1 border rounded dark:bg-gray-700 dark:border-gray-600"
              />
            </div>
          )}
          
          {param.type === 'boolean' && (
            <input
              type="checkbox"
              checked={localParams[param.name] || param.default}
              onChange={(e) => handleChange(param.name, e.target.checked)}
              className="rounded"
            />
          )}
        </div>
      ))}
      
      <button
        onClick={() => {
          const defaults = parameterDefs.reduce((acc, param) => ({
            ...acc,
            [param.name]: param.default
          }), {});
          setLocalParams(defaults);
          onParameterChange(defaults);
        }}
        className="w-full px-4 py-2 bg-gray-200 dark:bg-gray-700 rounded hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
      >
        Reset to Defaults
      </button>
    </div>
  );
}