/**
 * ExamPage — Core proctored exam view.
 *
 * Layout: Left 70% = questions, Right 30% = camera + status
 * Real-time pipeline:
 *   - Camera frames every 1.5s → WebSocket
 *   - Keystroke + mouse events every 2s → WebSocket
 *   - Backend responds with status/flag messages
 */
import { useState, useEffect, useCallback, useRef } from 'react'
import { useParams, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { examApi } from '../services/api'
import useWebSocket from '../hooks/useWebSocket'
import useKeystrokes from '../hooks/useKeystrokes'
import useMouseTracker from '../hooks/useMouseTracker'
import CameraFeed from '../components/CameraFeed'
import StatusIndicator from '../components/StatusIndicator'
import QuestionPanel from '../components/QuestionPanel'
import ExamTimer from '../components/ExamTimer'
import WarningBanner from '../components/WarningBanner'

export default function ExamPage() {
  const { sessionId } = useParams()
  const { state }     = useLocation()
  const { token }     = useAuth()
  const navigate      = useNavigate()

  const [questions, setQuestions]       = useState(state?.examData?.questions || [])
  const [examTitle, setExamTitle]       = useState(state?.examData?.exam_title || '')
  const [answers, setAnswers]           = useState({})
  const [cheatingProb, setCheatingProb] = useState(0)
  const [classLabel, setClassLabel]     = useState('Normal')
  const [flagCount, setFlagCount]       = useState(0)
  const [showWarning, setShowWarning]   = useState(false)
  const [warningMsg, setWarningMsg]     = useState('')
  const [submitting, setSubmitting]     = useState(false)

  const sid = parseInt(sessionId)

  const { isConnected, connectionState, lastStatus, lastFlag, sendFrame, sendBehavior, connect, disconnect } =
    useWebSocket(sid, token)
  const { getAndClearBuffer: getKeys }  = useKeystrokes()
  const { getAndClearBuffer: getMouse } = useMouseTracker()

  // Connect WebSocket on mount
  useEffect(() => {
    if (sid && token) connect()
    return () => disconnect()
  }, [sid, token])

  // Handle status updates
  useEffect(() => {
    if (!lastStatus) return
    setCheatingProb(lastStatus.cheating_probability)
    setClassLabel(lastStatus.class_name)
  }, [lastStatus])

  // Handle flag messages
  useEffect(() => {
    if (!lastFlag) return
    setFlagCount(c => c + 1)
    setWarningMsg(`${lastFlag.class_name} detected (${Math.round(lastFlag.cheating_probability * 100)}%)`)
    setShowWarning(true)
  }, [lastFlag])

  // Send behavioral data every 2s
  useEffect(() => {
    const t = setInterval(() => {
      const ks    = getKeys()
      const mouse = getMouse()
      if (ks.length || mouse.length) {
        sendBehavior(ks, mouse)
      }
    }, 2000)
    return () => clearInterval(t)
  }, [sendBehavior])

  // Handle answer selection + auto-save
  const handleAnswer = useCallback(async (questionId, answer) => {
    setAnswers(a => ({ ...a, [questionId]: answer }))
    try {
      await examApi.answer(sid, questionId, answer)
    } catch { /* ignore — will retry */ }
  }, [sid])

  // Submit exam
  const handleSubmit = async () => {
    if (!window.confirm('Submit your exam? This cannot be undone.')) return
    setSubmitting(true)
    try {
      await examApi.end(sid)
      disconnect()
      navigate(`/complete/${sid}`)
    } catch (err) {
      alert('Failed to submit exam. Please try again.')
      setSubmitting(false)
    }
  }

  const answeredCount = Object.keys(answers).length
  const statusColor = cheatingProb >= 0.7 ? 'text-red-500' : cheatingProb >= 0.5 ? 'text-yellow-500' : 'text-green-500'

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col">
      <WarningBanner message={warningMsg} visible={showWarning} onDismiss={() => setShowWarning(false)} />

      {/* Top bar */}
      <div className="bg-gray-900 text-white px-6 py-3 flex items-center justify-between text-sm shadow">
        <div className="flex items-center gap-4">
          <span className="font-bold text-indigo-400">InvigilAI</span>
          <span className="text-gray-400">{examTitle}</span>
        </div>
        <div className="flex items-center gap-6">
          <span className="text-gray-400">
            Answered: <strong className="text-white">{answeredCount}/{questions.length}</strong>
          </span>
          <span className={`text-xs font-semibold ${statusColor}`}>{classLabel}</span>
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'} animate-pulse`} />
          <ExamTimer durationSeconds={1800} onExpire={handleSubmit} />
        </div>
      </div>

      {/* Connection lost overlay */}
      {connectionState === 'disconnected' && (
        <div className="bg-yellow-50 border-b border-yellow-200 text-yellow-700 text-sm px-6 py-2 text-center">
          ⚠ Proctoring connection lost — attempting to reconnect…
        </div>
      )}

      {/* Main content */}
      <div className="flex flex-1 overflow-hidden p-4 gap-4 max-w-screen-xl mx-auto w-full">
        {/* Left — questions (70%) */}
        <div className="flex-1 min-w-0">
          <QuestionPanel questions={questions} answers={answers} onAnswer={handleAnswer} />
        </div>

        {/* Right — camera + status (30%) */}
        <div className="w-72 flex-shrink-0 flex flex-col gap-4">
          {/* Camera */}
          <CameraFeed onFrame={sendFrame} captureInterval={1500} className="aspect-video w-full" />

          {/* Status card */}
          <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm text-center">
            <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">Proctoring Status</div>
            <StatusIndicator probability={cheatingProb} />
            <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-gray-400">
              <div className="bg-gray-50 rounded-lg p-2">
                <div className="font-bold text-gray-700">{flagCount}</div>
                <div>Flags</div>
              </div>
              <div className="bg-gray-50 rounded-lg p-2">
                <div className="font-bold text-gray-700">{Math.round(cheatingProb * 100)}%</div>
                <div>Suspicion</div>
              </div>
            </div>
          </div>

          {/* Submit */}
          <button
            onClick={handleSubmit}
            disabled={submitting}
            className="w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-3 rounded-xl text-sm shadow disabled:opacity-40"
          >
            {submitting ? 'Submitting…' : '✓ Submit Exam'}
          </button>

          <p className="text-xs text-gray-400 text-center">
            {answeredCount} of {questions.length} questions answered
          </p>
        </div>
      </div>
    </div>
  )
}
