import React from 'react';

interface LogoProps {
  className?: string;
  showText?: boolean;
  animated?: boolean;
  width?: number;
}

export function Logo({ className = "", showText = true, animated = false, width = 300 }: LogoProps) {
  const scale = width / 300;
  const height = 60 * scale;

  return (
    <svg 
      width={width} 
      height={height} 
      viewBox="0 0 300 60"
      xmlns="http://www.w3.org/2000/svg" 
      role="img"
      aria-labelledby="vtaTitle vtaDesc"
      className={className}
    >
      <title id="vtaTitle">VectorTA logo</title>
      <desc id="vtaDesc">
        Transformer-style V with input/output vector bars and the word 'ectorTA'.
      </desc>

      {/* Gradient palette */}
      <defs>
        <linearGradient id="vtaGradient" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="rgb(59, 130, 246)"/>
          <stop offset="1" stopColor="rgb(139, 92, 246)"/>
        </linearGradient>

        {animated && (
          <style>{`
            /* ===== Accessibility ===== */
            @media (prefers-reduced-motion: reduce) {
              .anim { animation: none !important; }
            }
            @media (prefers-contrast: more) {
              .v-flare, .output-bar { fill: rgb(23, 23, 23) !important; }
            }

            /* ===== Animations ===== */
            @keyframes inShift {
              0%   { transform: translateX(0);   opacity: 1; }
              45%  { transform: translateX(18px); opacity: 1; }
              100% { transform: translateX(18px); opacity: 0; }
            }
            @keyframes outShift {
              0%   { transform: translateX(-18px); opacity: 0; }
              55%  { transform: translateX(-18px); opacity: 1; }
              100% { transform: translateX(0);     opacity: 1; }
            }
            @keyframes flare {
              0%,100% { fill: url(#vtaGradient); }
              50%     { fill: rgb(59, 130, 246); }
            }

            .input-bar  { animation: inShift 1.2s ease-in-out infinite; }
            .output-bar { animation: outShift 1.2s ease-in-out infinite; }
            .v-flare    { animation: flare   2.4s ease-in-out infinite; }
          `}</style>
        )}
      </defs>

      {/* Icon block */}
      <g id="vta-icon" transform="translate(0 6)">
        {/* Input bars */}
        <g>
          <rect 
            className={animated ? "input-bar anim" : ""} 
            x="2" y="8" width="8" height="2"
            fill="rgb(100, 116, 139)"
            style={animated ? {animationDelay: "0s"} : undefined}
          />
          <rect 
            className={animated ? "input-bar anim" : ""} 
            x="2" y="14" width="8" height="2"
            fill="rgb(100, 116, 139)"
            style={animated ? {animationDelay: "0.08s"} : undefined}
          />
          <rect 
            className={animated ? "input-bar anim" : ""} 
            x="2" y="20" width="8" height="2"
            fill="rgb(100, 116, 139)"
            style={animated ? {animationDelay: "0.16s"} : undefined}
          />
          <rect 
            className={animated ? "input-bar anim" : ""} 
            x="2" y="26" width="8" height="2"
            fill="rgb(100, 116, 139)"
            style={animated ? {animationDelay: "0.24s"} : undefined}
          />
        </g>

        {/* Transformer V */}
        <text 
          className={animated ? "v-flare anim" : ""} 
          x="20" y="28" 
          textAnchor="middle"
          fontFamily="Inter, sans-serif" 
          fontSize="28" 
          fontWeight="700"
          fill="url(#vtaGradient)"
        >
          V
        </text>

        {/* Output bars */}
        <g>
          <rect 
            className={animated ? "output-bar anim" : ""} 
            x="30" y="8" width="8" height="2"
            fill="url(#vtaGradient)" 
            style={animated ? {animationDelay: "0.24s"} : undefined}
          />
          <rect 
            className={animated ? "output-bar anim" : ""} 
            x="30" y="14" width="8" height="2"
            fill="url(#vtaGradient)" 
            style={animated ? {animationDelay: "0.16s"} : undefined}
          />
          <rect 
            className={animated ? "output-bar anim" : ""} 
            x="30" y="20" width="8" height="2"
            fill="url(#vtaGradient)" 
            style={animated ? {animationDelay: "0.08s"} : undefined}
          />
          <rect 
            className={animated ? "output-bar anim" : ""} 
            x="30" y="26" width="8" height="2"
            fill="url(#vtaGradient)" 
            style={animated ? {animationDelay: "0s"} : undefined}
          />
        </g>
      </g>

      {/* Wordmark */}
      {showText && (
        <text x="40" y="34" fontFamily="Inter, sans-serif" fontSize="28">
          <tspan fill="currentColor" fontWeight="600">ector</tspan>
          <tspan fill="url(#vtaGradient)" fontWeight="700">TA</tspan>
        </text>
      )}
    </svg>
  );
}