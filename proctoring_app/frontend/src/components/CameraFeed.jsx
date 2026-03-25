/**
 * CameraFeed — Live webcam preview that captures frames for the WebSocket.
 *
 * Props:
 *   onFrame(base64) — called every captureInterval ms with a JPEG frame
 *   captureInterval — ms between frames (default 1500)
 *   className
 */
import { useEffect, useRef } from 'react'
import useCamera from '../hooks/useCamera'

export default function CameraFeed({ onFrame, captureInterval = 1500, className = '' }) {
  const { videoRef, canvasRef, isActive, error, startCamera, captureFrame } = useCamera()

  // Start camera on mount
  useEffect(() => { startCamera() }, [])

  // Send frames on interval
  useEffect(() => {
    if (!isActive || !onFrame) return
    const t = setInterval(() => {
      const frame = captureFrame()
      if (frame) onFrame(frame)
    }, captureInterval)
    return () => clearInterval(t)
  }, [isActive, captureInterval])

  return (
    <div className={`relative bg-black rounded-lg overflow-hidden ${className}`}>
      {error ? (
        <div className="flex items-center justify-center h-full min-h-40 text-red-400 text-sm p-4 text-center">
          {error}
        </div>
      ) : (
        <>
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            className="w-full h-full object-cover"
          />
          {!isActive && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/70 text-white text-sm">
              Starting camera…
            </div>
          )}
        </>
      )}
      {/* Recording indicator */}
      {isActive && (
        <div className="absolute top-2 right-2 flex items-center gap-1">
          <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
          <span className="text-white text-xs">REC</span>
        </div>
      )}
      {/* Hidden canvas for frame capture */}
      <canvas ref={canvasRef} className="hidden" />
    </div>
  )
}
