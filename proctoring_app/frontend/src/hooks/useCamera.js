/**
 * useCamera — Access the webcam and provide a captureFrame() utility.
 *
 * Returns:
 *   videoRef   — ref to attach to <video> element
 *   canvasRef  — hidden canvas used for frame capture
 *   isActive   — true once getUserMedia succeeds
 *   error      — error message string or null
 *   captureFrame() — returns base64 JPEG string of current video frame
 *   stopCamera()   — stops all tracks
 */

import { useRef, useState, useCallback, useEffect } from 'react'

export default function useCamera() {
  const videoRef  = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const [isActive, setIsActive] = useState(false)
  const [error, setError]       = useState(null)

  const startCamera = useCallback(async () => {
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        try {
          await videoRef.current.play()
        } catch (playErr) {
          // Ignore AbortError — happens when component re-renders mid-play
          if (playErr.name !== 'AbortError') throw playErr
        }
      }
      setIsActive(true)
    } catch (err) {
      setIsActive(false)
      if (err.name === 'NotAllowedError') {
        setError('Camera permission denied. Please allow camera access and refresh.')
      } else if (err.name === 'NotFoundError') {
        setError('No camera detected. Please connect a webcam.')
      } else {
        setError(`Camera error: ${err.message}`)
      }
    }
  }, [])

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop())
    setIsActive(false)
  }, [])

  const captureFrame = useCallback(() => {
    const video  = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || !isActive) return null

    canvas.width  = video.videoWidth  || 640
    canvas.height = video.videoHeight || 480
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    return canvas.toDataURL('image/jpeg', 0.7)
  }, [isActive])

  // Auto-cleanup on unmount
  useEffect(() => () => stopCamera(), [stopCamera])

  return { videoRef, canvasRef, isActive, error, startCamera, stopCamera, captureFrame }
}
