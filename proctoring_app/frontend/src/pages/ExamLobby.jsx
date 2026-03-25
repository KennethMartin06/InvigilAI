/**
 * ExamLobby — Pre-exam page with camera check and exam selection.
 * Students can only start the exam when the camera is active.
 */
import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import Navbar from '../components/Navbar'
import { examApi } from '../services/api'
import useCamera from '../hooks/useCamera'

export default function ExamLobby() {
  const navigate = useNavigate()
  const { videoRef, canvasRef, isActive, error, startCamera } = useCamera()
  const [examTitle, setExamTitle]   = useState('General Knowledge')
  const [loading, setLoading]       = useState(false)
  const [startError, setStartError] = useState(null)

  useEffect(() => { startCamera() }, [])

  const handleStart = async () => {
    if (!isActive) { setStartError('Please allow camera access first.'); return }
    setLoading(true)
    setStartError(null)
    try {
      const res = await examApi.start(examTitle)
      const { session_id } = res.data
      navigate(`/exam/${session_id}`, { state: { examData: res.data } })
    } catch (err) {
      setStartError(err.response?.data?.detail || 'Failed to start exam.')
      setLoading(false)
    }
  }

  const checks = [
    { label: 'Camera access',         ok: isActive,  note: error || 'Required for proctoring' },
    { label: 'Browser compatibility', ok: true,       note: 'WebSocket & Canvas supported' },
    { label: 'Exam selected',         ok: !!examTitle, note: examTitle },
  ]

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />

      <div className="max-w-4xl mx-auto px-4 py-10">
        <h1 className="text-2xl font-bold text-gray-800 mb-2">Exam Lobby</h1>
        <p className="text-gray-500 text-sm mb-8">
          Complete the checks below before starting your proctored exam.
        </p>

        <div className="grid md:grid-cols-2 gap-8">
          {/* Camera preview */}
          <div>
            <h2 className="text-sm font-semibold text-gray-600 uppercase tracking-wide mb-3">Camera Preview</h2>
            <div className="relative bg-black rounded-xl overflow-hidden aspect-video shadow-lg">
              {error ? (
                <div className="flex items-center justify-center h-full text-red-400 text-sm p-4 text-center">{error}</div>
              ) : (
                <video ref={videoRef} autoPlay muted playsInline className="w-full h-full object-cover" />
              )}
              {isActive && (
                <div className="absolute top-3 right-3 bg-green-500 text-white text-xs px-2 py-0.5 rounded-full">
                  ✓ Live
                </div>
              )}
            </div>
            <canvas ref={canvasRef} className="hidden" />
            <p className="text-xs text-gray-400 mt-2 text-center">
              Your camera must remain visible throughout the exam.
            </p>
          </div>

          {/* Checklist + start */}
          <div className="space-y-5">
            <div>
              <h2 className="text-sm font-semibold text-gray-600 uppercase tracking-wide mb-3">System Checklist</h2>
              <div className="bg-white rounded-xl border border-gray-200 divide-y">
                {checks.map(c => (
                  <div key={c.label} className="flex items-center gap-3 px-4 py-3">
                    <div className={`w-5 h-5 rounded-full flex items-center justify-center text-white text-xs flex-shrink-0
                      ${c.ok ? 'bg-green-500' : 'bg-gray-200'}`}>
                      {c.ok ? '✓' : '○'}
                    </div>
                    <div>
                      <div className="text-sm font-medium text-gray-700">{c.label}</div>
                      <div className={`text-xs ${c.ok ? 'text-gray-400' : 'text-red-500'}`}>{c.note}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Select Exam</label>
              <select
                value={examTitle}
                onChange={e => setExamTitle(e.target.value)}
                className="w-full border border-gray-300 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                <option>General Knowledge</option>
              </select>
            </div>

            {startError && (
              <div className="bg-red-50 border border-red-200 text-red-600 text-sm rounded-lg px-4 py-3">
                {startError}
              </div>
            )}

            <button
              onClick={handleStart}
              disabled={!isActive || loading}
              className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 rounded-xl text-sm disabled:opacity-40 shadow-md"
            >
              {loading ? 'Starting…' : '▶ Start Proctored Exam'}
            </button>

            <p className="text-xs text-gray-400 text-center">
              By starting the exam you consent to AI-based monitoring of your webcam and keyboard activity.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
