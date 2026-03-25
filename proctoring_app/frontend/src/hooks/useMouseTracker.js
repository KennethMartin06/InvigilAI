/**
 * useMouseTracker — Capture mousemove (throttled) and click events.
 *
 * Returns:
 *   getAndClearBuffer() — returns current event array and resets it
 */

import { useRef, useEffect, useCallback } from 'react'

const THROTTLE_MS = 100

export default function useMouseTracker() {
  const buffer  = useRef([])
  const lastMov = useRef(0)

  useEffect(() => {
    const onMove = (e) => {
      const now = performance.now()
      if (now - lastMov.current < THROTTLE_MS) return
      lastMov.current = now
      buffer.current.push({ x: e.clientX, y: e.clientY, type: 'move', timestamp: now })
    }
    const onClick = (e) => {
      buffer.current.push({ x: e.clientX, y: e.clientY, type: 'click', timestamp: performance.now() })
    }
    document.addEventListener('mousemove', onMove)
    document.addEventListener('click', onClick)
    return () => {
      document.removeEventListener('mousemove', onMove)
      document.removeEventListener('click', onClick)
    }
  }, [])

  const getAndClearBuffer = useCallback(() => {
    const events = [...buffer.current]
    buffer.current = []
    return events
  }, [])

  return { getAndClearBuffer }
}
