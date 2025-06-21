import React from 'react';

interface LogoProps {
  className?: string;
  width?: number;
}

export function LogoSimple({ className = "", width = 180 }: LogoProps) {
  return (
    <div className={`flex items-center gap-2 ${className}`}>
      {/* Simple V icon */}
      <div className="flex items-center justify-center w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg">
        <span className="text-white font-bold text-xl">V</span>
      </div>
      
      {/* Text */}
      <div className="flex items-baseline">
        <span className="text-2xl font-semibold text-foreground">ector</span>
        <span className="text-2xl font-bold bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">TA</span>
      </div>
    </div>
  );
}