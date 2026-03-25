/**
 * StatusIndicator — Coloured badge showing real-time cheating probability.
 * Green = Normal (<0.5), Yellow = Suspicious (0.5–0.7), Red = Cheating (>0.7)
 */
import { statusHex, statusLabel } from '../utils/featureHelpers'

export default function StatusIndicator({ probability = 0, className = '' }) {
  const colour = statusHex(probability)
  const label  = statusLabel(probability)
  const pct    = Math.round(probability * 100)

  return (
    <div className={`flex flex-col items-center gap-1 ${className}`}>
      <div
        className="w-4 h-4 rounded-full shadow-lg animate-pulse"
        style={{ backgroundColor: colour, boxShadow: `0 0 8px ${colour}` }}
      />
      <span className="text-xs font-semibold" style={{ color: colour }}>
        {label}
      </span>
      <span className="text-xs text-gray-400">{pct}% cheating prob.</span>
    </div>
  )
}
