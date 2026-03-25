/**
 * LiveSessionCard — Compact card showing a live exam session for admin view.
 *
 * Props:
 *   session    — SessionResponse object
 *   onClick()  — navigate to session review
 */
import { statusHex, statusLabel } from '../utils/featureHelpers'

function elapsed(startTime) {
  const ms = Date.now() - new Date(startTime).getTime()
  const m = Math.floor(ms / 60000)
  const s = Math.floor((ms % 60000) / 1000)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

export default function LiveSessionCard({ session, onClick }) {
  const colour = statusHex(session.max_cheating_score)
  const label  = statusLabel(session.max_cheating_score)

  return (
    <div
      onClick={onClick}
      className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm hover:shadow-md cursor-pointer hover:border-indigo-300 transition"
    >
      <div className="flex items-start justify-between mb-3">
        <div>
          <div className="font-semibold text-gray-800 text-sm">{session.student_name || `User #${session.user_id}`}</div>
          <div className="text-xs text-gray-400">{session.student_email}</div>
        </div>
        <div
          className="w-3 h-3 rounded-full animate-pulse mt-1"
          style={{ backgroundColor: colour, boxShadow: `0 0 6px ${colour}` }}
        />
      </div>

      <div className="text-xs text-gray-500 mb-2 truncate">{session.exam_title}</div>

      <div className="flex items-center justify-between text-xs">
        <span className="text-gray-400">⏱ {elapsed(session.start_time)}</span>
        <span className="font-medium" style={{ color: colour }}>{label}</span>
      </div>

      <div className="flex items-center justify-between mt-2 text-xs">
        <span className="text-gray-500">🚩 {session.total_flags} flags</span>
        <span className="text-gray-500">Peak: {Math.round(session.max_cheating_score * 100)}%</span>
      </div>
    </div>
  )
}
