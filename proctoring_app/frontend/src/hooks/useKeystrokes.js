/**
 * useKeystrokes — Capture keydown/keyup events and buffer them per window.
 *
 * Returns:
 *   getAndClearBuffer() — returns current event array and resets it
 */

import { useRef, useEffect, useCallback } from 'react'

export default function useKeystrokes() {
  const buffer = useRef([])

  useEffect(() => {
    const onDown = (e) => {
      buffer.current.push({ key: e.key, type: 'down', timestamp: performance.now() })
    }
    const onUp = (e) => {
      buffer.current.push({ key: e.key, type: 'up', timestamp: performance.now() })
    }
    document.addEventListener('keydown', onDown)
    document.addEventListener('keyup', onUp)
    return () => {
      document.removeEventListener('keydown', onDown)
      document.removeEventListener('keyup', onUp)
    }
  }, [])

  const getAndClearBuffer = useCallback(() => {
    const events = [...buffer.current]
    buffer.current = []
    return events
  }, [])

  return { getAndClearBuffer }
}
