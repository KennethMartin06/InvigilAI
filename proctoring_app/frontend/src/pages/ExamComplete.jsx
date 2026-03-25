/** ExamComplete — Post-submission summary page for students. */
import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { examApi, flagApi } from '../services/api'
import Navbar from '../components/Navbar'

export default function ExamComplete() {
  const { sessionId } = useParams()
  const navigate = useNavigate()
  const [summary, setSummary] = useState(null)

  useEffect(() => {
    examApi.end(parseInt(sessionId))
      .then(r => setSummary(r.data))
      .catch(() => {})
  }, [sessionId])

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      <div className="max-w-lg mx-auto px-4 py-16 text-center">
        <div className="text-6xl mb-4">✅</div>
        <h1 className="text-2xl font-bold text-gray-800 mb-2">Exam Submitted!</h1>
        <p className="text-gray-500 mb-8">Your responses have been recorded successfully.</p>

        {summary && (
          <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-6 text-left space-y-3 mb-8">
            <Row label="Exam"            value={summary.exam_title} />
            <Row label="Duration"        value={`${Math.round(summary.duration_seconds / 60)} min ${summary.duration_seconds % 60} sec`} />
            <Row label="Questions"       value={`${summary.answered_questions} / ${summary.total_questions} answered`} />
            <Row label="Flags received"  value={summary.total_flags} highlight={summary.total_flags > 0} />
            <Row label="Status"          value={summary.status} />
          </div>
        )}

        <button
          onClick={() => navigate('/lobby')}
          className="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold px-6 py-3 rounded-xl text-sm shadow"
        >
          Back to Lobby
        </button>
      </div>
    </div>
  )
}

function Row({ label, value, highlight = false }) {
  return (
    <div className="flex justify-between text-sm">
      <span className="text-gray-500">{label}</span>
      <span className={`font-semibold ${highlight ? 'text-red-600' : 'text-gray-800'}`}>{value}</span>
    </div>
  )
}
