/**
 * ExamTimer — Countdown timer with warning when < 5 minutes remain.
 * Calls onExpire() when it reaches zero.
 */
import { useEffect, useState } from 'react'
import { formatTime } from '../utils/featureHelpers'

export default function ExamTimer({ durationSeconds = 1800, onExpire }) {
  const [remaining, setRemaining] = useState(durationSeconds)

  useEffect(() => {
    if (remaining <= 0) { onExpire?.(); return }
    const t = setTimeout(() => setRemaining(r => r - 1), 1000)
    return () => clearTimeout(t)
  }, [remaining])

  const warning = remaining <= 300  // last 5 minutes
  const colour  = remaining <= 60 ? 'text-red-500' : warning ? 'text-yellow-500' : 'text-gray-800'

  return (
    <div className={`text-center font-mono text-2xl font-bold ${colour}`}>
      {formatTime(remaining)}
      {warning && <div className="text-xs font-normal text-yellow-600">⏱ Time running out!</div>}
    </div>
  )
}
