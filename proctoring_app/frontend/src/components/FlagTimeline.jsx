/**
 * FlagTimeline — Vertical chronological list of cheating flags with thumbnails.
 *
 * Props:
 *   flags      — array of flag objects from API
 *   onAcknowledge(flagId) — callback for admin acknowledge action
 *   isAdmin    — show acknowledge button if true
 */
import { useState } from 'react'
import { statusHex } from '../utils/featureHelpers'

function FlagCard({ flag, isAdmin, onAcknowledge }) {
  const [expanded, setExpanded] = useState(false)
  const colour = statusHex(flag.cheating_probability)

  return (
    <div className={`relative pl-5 pb-6 ${flag.acknowledged ? 'opacity-60' : ''}`}>
      {/* Timeline dot */}
      <div
        className="absolute left-0 top-1 w-3 h-3 rounded-full border-2 border-white shadow"
        style={{ backgroundColor: colour }}
      />
      {/* Timeline line */}
      <div className="absolute left-1.5 top-4 bottom-0 w-px bg-gray-200" />

      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <span className="font-semibold text-sm" style={{ color: colour }}>
              {flag.class_name}
            </span>
            <span className="text-xs text-gray-400 ml-2">
              {new Date(flag.timestamp).toLocaleTimeString()}
            </span>
            <div className="text-xs text-gray-500 mt-0.5">
              Probability: {Math.round(flag.cheating_probability * 100)}%
            </div>
          </div>

          <div className="flex gap-2">
            {flag.screenshot_path && (
              <button
                onClick={() => setExpanded(e => !e)}
                className="text-xs text-indigo-600 hover:underline"
              >
                {expanded ? 'Hide' : 'Screenshot'}
              </button>
            )}
            {isAdmin && !flag.acknowledged && (
              <button
                onClick={() => onAcknowledge?.(flag.id)}
                className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded hover:bg-green-200"
              >
                Acknowledge
              </button>
            )}
            {flag.acknowledged && (
              <span className="text-xs text-green-600 font-medium">✓ Reviewed</span>
            )}
          </div>
        </div>

        {expanded && flag.screenshot_path && (
          <img
            src={`/uploads/${flag.screenshot_path.split('/uploads/')[1] || flag.screenshot_path}`}
            alt="screenshot"
            className="mt-3 rounded border max-h-40 object-contain"
          />
        )}

        <button
          onClick={() => setExpanded(e => !e)}
          className="text-xs text-gray-400 mt-2 hover:text-gray-600"
        >
          {expanded ? '▲ Hide details' : '▼ Show features'}
        </button>

        {expanded && flag.visual_features && (
          <div className="mt-2 grid grid-cols-2 gap-x-4 text-xs text-gray-500">
            {Object.entries(flag.visual_features).map(([k, v]) => (
              <div key={k} className="flex justify-between">
                <span>{k}:</span>
                <span className="font-mono">{typeof v === 'number' ? v.toFixed(3) : v}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default function FlagTimeline({ flags = [], isAdmin = false, onAcknowledge }) {
  if (!flags.length) {
    return <div className="text-gray-400 text-sm text-center py-8">No flags recorded for this session.</div>
  }

  return (
    <div className="relative">
      {flags.map(flag => (
        <FlagCard key={flag.id} flag={flag} isAdmin={isAdmin} onAcknowledge={onAcknowledge} />
      ))}
    </div>
  )
}
