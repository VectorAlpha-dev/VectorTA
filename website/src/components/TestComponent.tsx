import { useState, useEffect } from 'react';

export function TestComponent() {
  const [count, setCount] = useState(0);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    console.log('TestComponent mounted');
  }, []);

  return (
    <div className="p-4 border-2 border-blue-500 rounded-lg">
      <h3 className="text-xl font-bold mb-2">React Test Component</h3>
      <p>Mounted: {mounted ? 'Yes' : 'No'}</p>
      <p>Count: {count}</p>
      <button 
        onClick={() => setCount(count + 1)}
        className="mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
      >
        Increment
      </button>
    </div>
  );
}