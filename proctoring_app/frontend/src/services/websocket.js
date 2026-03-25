/**
 * websocket.js — WebSocket client class for real-time proctoring.
 *
 * Usage:
 *   const ws = new ProctorSocket(sessionId, token, { onStatus, onFlag, onError })
 *   ws.connect()
 *   ws.sendFrame(base64)
 *   ws.sendBehavior(keystrokes, mouse)
 *   ws.close()
 */

const WS_BASE = import.meta.env.VITE_WS_BASE || 'ws://localhost:8000'
const MAX_RETRIES = 3

export class ProctorSocket {
  /**
   * @param {number}   sessionId   - ExamSession ID
   * @param {string}   token       - JWT access token
   * @param {object}   handlers    - { onStatus, onFlag, onError, onConnect, onDisconnect }
   */
  constructor(sessionId, token, handlers = {}) {
    this.sessionId = sessionId
    this.token = token
    this.handlers = handlers
    this._ws = null
    this._retries = 0
    this._closed = false
  }

  connect() {
    if (this._closed) return
    const url = `${WS_BASE}/ws/proctor/${this.sessionId}?token=${this.token}`
    this._ws = new WebSocket(url)

    this._ws.onopen = () => {
      this._retries = 0
      this.handlers.onConnect?.()
    }

    this._ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)
        if (msg.type === 'status') this.handlers.onStatus?.(msg)
        else if (msg.type === 'flag')   this.handlers.onFlag?.(msg)
        else if (msg.type === 'error')  this.handlers.onError?.(msg.message)
      } catch (e) {
        // ignore malformed
      }
    }

    this._ws.onclose = (event) => {
      this.handlers.onDisconnect?.()
      if (!this._closed && this._retries < MAX_RETRIES) {
        const delay = Math.pow(2, this._retries) * 1000  // 1s, 2s, 4s
        this._retries++
        setTimeout(() => this.connect(), delay)
      }
    }

    this._ws.onerror = () => {
      this.handlers.onError?.('WebSocket connection error')
    }
  }

  sendFrame(base64) {
    if (this._ws?.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify({ type: 'frame', data: base64 }))
    }
  }

  sendBehavior(keystrokes, mouse) {
    if (this._ws?.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify({ type: 'behavior', keystrokes, mouse }))
    }
  }

  get isConnected() {
    return this._ws?.readyState === WebSocket.OPEN
  }

  close() {
    this._closed = true
    this._ws?.close()
  }
}
