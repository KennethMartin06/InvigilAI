/**
 * WarningBanner — Full-width alert that slides in when cheating is flagged.
 * Auto-dismisses after 5 seconds.
 */
import { useEffect, useState } from 'react'

export default function WarningBanner({ message, visible, onDismiss }) {
  const [show, setShow] = useState(false)

  useEffect(() => {
    if (visible) {
      setShow(true)
      const t = setTimeout(() => {
        setShow(false)
        onDismiss?.()
      }, 5000)
      return () => clearTimeout(t)
    }
  }, [visible, message])

  if (!show) return null

  return (
    <div className="fixed top-0 left-0 right-0 z-50 animate-bounce">
      <div className="bg-red-600 text-white px-6 py-3 flex items-center justify-between shadow-xl">
        <span className="font-semibold text-sm">
          ⚠ {message || 'Suspicious activity detected. Please focus on your screen.'}
        </span>
        <button onClick={() => { setShow(false); onDismiss?.() }} className="text-white text-lg leading-none">
          ×
        </button>
      </div>
    </div>
  )
}
