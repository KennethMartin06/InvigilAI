/**
 * featureHelpers.js — Client-side behavioral feature helpers.
 *
 * These functions compute the same metrics as the backend behavior_processor
 * for validation / debugging purposes. The actual inference always runs
 * server-side. The frontend only needs to send raw event buffers.
 */

/** Compute keys per second from a buffer of keydown events. */
export function computeKeystrokeRate(keydownEvents, windowSec = 2.0) {
  return keydownEvents.length / windowSec
}

/** Mean key hold duration in ms. */
export function computeMeanDwellTime(keydownEvents, keyupEvents) {
  const upMap = {}
  keyupEvents.forEach(e => { upMap[e.key] = e.timestamp })
  const dwells = keydownEvents
    .map(e => (upMap[e.key] != null ? upMap[e.key] - e.timestamp : null))
    .filter(d => d != null && d > 0 && d < 2000)
  if (!dwells.length) return 100
  return dwells.reduce((a, b) => a + b, 0) / dwells.length
}

/** Mean gap between consecutive keydown events in ms. */
export function computeMeanFlightTime(keydownEvents) {
  const sorted = [...keydownEvents].sort((a, b) => a.timestamp - b.timestamp)
  if (sorted.length < 2) return 150
  const intervals = []
  for (let i = 1; i < sorted.length; i++) {
    const gap = sorted[i].timestamp - sorted[i - 1].timestamp
    if (gap > 0 && gap < 5000) intervals.push(gap)
  }
  if (!intervals.length) return 150
  return intervals.reduce((a, b) => a + b, 0) / intervals.length
}

/** Format seconds into MM:SS display string. */
export function formatTime(seconds) {
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

/** Return a CSS class string for a cheating probability score. */
export function statusColor(probability) {
  if (probability >= 0.7) return 'text-red-500'
  if (probability >= 0.5) return 'text-yellow-500'
  return 'text-green-500'
}

/** Return a hex colour for a cheating probability score. */
export function statusHex(probability) {
  if (probability >= 0.7) return '#ef4444'
  if (probability >= 0.5) return '#eab308'
  return '#22c55e'
}

/** Return a label for a cheating probability score. */
export function statusLabel(probability) {
  if (probability >= 0.7) return 'Cheating Detected'
  if (probability >= 0.5) return 'Suspicious'
  return 'Normal'
}
