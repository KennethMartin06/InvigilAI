/**
 * useWebSocket — Manages a ProctorSocket connection lifecycle.
 *
 * Returns:
 *   isConnected     — boolean
 *   connectionState — 'connecting' | 'connected' | 'disconnected' | 'error'
 *   lastStatus      — latest status message object
 *   lastFlag        — latest flag message object
 *   sendFrame(b64)  — send a camera frame
 *   sendBehavior(ks, mouse) — send behavioral event batch
 *   connect()       — open / re-open connection
 *   disconnect()    — close permanently
 */

import { useState, useRef, useCallback, useEffect } from 'react'
import { ProctorSocket } from '../services/websocket'

export default function useWebSocket(sessionId, token) {
  const [connectionState, setConnectionState] = useState('disconnected')
  const [lastStatus, setLastStatus]           = useState(null)
  const [lastFlag, setLastFlag]               = useState(null)
  const socketRef = useRef(null)

  const connect = useCallback(() => {
    if (!sessionId || !token) return

    socketRef.current?.close()
    setConnectionState('connecting')

    const sock = new ProctorSocket(sessionId, token, {
      onConnect:    () => setConnectionState('connected'),
      onDisconnect: () => setConnectionState('disconnected'),
      onStatus:     (msg) => setLastStatus(msg),
      onFlag:       (msg) => setLastFlag(msg),
      onError:      ()    => setConnectionState('error'),
    })

    sock.connect()
    socketRef.current = sock
  }, [sessionId, token])

  const disconnect = useCallback(() => {
    socketRef.current?.close()
    socketRef.current = null
    setConnectionState('disconnected')
  }, [])

  const sendFrame = useCallback((b64) => {
    socketRef.current?.sendFrame(b64)
  }, [])

  const sendBehavior = useCallback((keystrokes, mouse) => {
    socketRef.current?.sendBehavior(keystrokes, mouse)
  }, [])

  // Cleanup on unmount
  useEffect(() => () => socketRef.current?.close(), [])

  return {
    isConnected: connectionState === 'connected',
    connectionState,
    lastStatus,
    lastFlag,
    sendFrame,
    sendBehavior,
    connect,
    disconnect,
  }
}
