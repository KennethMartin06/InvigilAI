/**
 * SessionReview — Detailed admin view for a single exam session.
 *
 * Shows: session summary, cheating probability timeline chart,
 *        flag distribution bar chart, and the full flag timeline.
 */
import { useState, useEffect, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import Navbar from '../components/Navbar'
import FlagTimeline from '../components/FlagTimeline'
import { adminApi, flagApi } from '../services/api'
import { statusHex } from '../utils/featureHelpers'

export default function SessionReview() {
  const { sessionId } = useParams()
  const navigate = useNavigate()
  const sid = parseInt(sessionId)

  const [sessions, setSessions] = useState([])
  const [session, setSession]   = useState(null)
  const [flags, setFlags]       = useState([])
  const [loading, setLoading]   = useState(true)
  const chartRef = useRef(null)

  useEffect(() => {
    async function load() {
      try {
        const [sRes, fRes] = await Promise.all([
          adminApi.sessions({ student_id: undefined }),
          flagApi.bySession(sid),
        ])
        // Find this specific session
        const all = sRes.data
        const found = all.find(s => s.id === sid)
        setSession(found || null)
        setFlags(fRes.data)
      } catch (err) {
        console.error('SessionReview load error:', err)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [sid])

  const handleAcknowledge = async (flagId) => {
    await flagApi.acknowledge(flagId)
    setFlags(f => f.map(fl => fl.id === flagId ? { ...fl, acknowledged: true } : fl))
  }

  // Inline mini-chart: cheating probability over flag timestamps
  const chartData = flags.map(f => ({
    t: new Date(f.timestamp).toLocaleTimeString(),
    p: Math.round(f.cheating_probability * 100),
  }))

  const flagsByClass = flags.reduce((acc, f) => {
    acc[f.class_name] = (acc[f.class_name] || 0) + 1
    return acc
  }, {})

  const topClass = Object.entries(flagsByClass).sort((a, b) => b[1] - a[1])[0]?.[0] || 'None'

  const duration = session?.end_time
    ? Math.round((new Date(session.end_time) - new Date(session.start_time)) / 60)
    : null

  if (loading) return (
    <div className="min-h-screen bg-gray-50"><Navbar />
      <div className="flex items-center justify-center h-64 text-gray-400">Loading session…</div>
    </div>
  )

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      <div className="max-w-screen-xl mx-auto px-6 py-8 space-y-8">

        {/* Breadcrumb */}
        <div className="flex items-center gap-2 text-sm">
          <button onClick={() => navigate('/admin')} className="text-indigo-600 hover:underline">Dashboard</button>
          <span className="text-gray-300">/</span>
          <span className="text-gray-500">Session #{sid}</span>
        </div>

        {/* Session header */}
        {session ? (
          <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-6">
            <div className="flex items-start justify-between">
              <div>
                <h1 className="text-xl font-bold text-gray-800">{session.student_name || `Student #${session.user_id}`}</h1>
                <p className="text-gray-400 text-sm">{session.student_email}</p>
                <p className="text-sm text-gray-600 mt-1">{session.exam_title}</p>
              </div>
              <span className={`text-xs font-semibold px-3 py-1.5 rounded-full
                ${session.status === 'active' ? 'bg-green-100 text-green-700'
                : session.status === 'completed' ? 'bg-blue-100 text-blue-700'
                : 'bg-red-100 text-red-700'}`}>
                {session.status}
              </span>
            </div>

            {/* Stats row */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-6">
              {[
                { label: 'Total Flags',   value: session.total_flags, colour: '#ef4444' },
                { label: 'Peak Score',    value: `${Math.round(session.max_cheating_score * 100)}%`, colour: statusHex(session.max_cheating_score) },
                { label: 'Duration',      value: duration != null ? `${duration} min` : '—', colour: '#6366f1' },
                { label: 'Most Common',   value: topClass, colour: '#f59e0b' },
              ].map(c => (
                <div key={c.label} className="bg-gray-50 rounded-xl p-4 border border-gray-100">
                  <div className="text-lg font-bold" style={{ color: c.colour }}>{c.value}</div>
                  <div className="text-xs text-gray-500 mt-0.5">{c.label}</div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="text-gray-400 text-sm">Session not found.</div>
        )}

        {/* Charts row */}
        {flags.length > 0 && (
          <div className="grid md:grid-cols-2 gap-6">
            {/* Probability timeline (simple SVG) */}
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-5">
              <h3 className="text-sm font-semibold text-gray-600 mb-4">Cheating Probability Over Time</h3>
              <svg width="100%" height="160" viewBox={`0 0 ${Math.max(chartData.length * 40, 200)} 120`}>
                {/* Threshold line */}
                <line x1="0" y1="42" x2={Math.max(chartData.length * 40, 200)} y2="42"
                  stroke="#ef4444" strokeDasharray="4,2" strokeWidth="1" opacity="0.5" />
                {/* Data line */}
                {chartData.length > 1 && (
                  <polyline
                    points={chartData.map((d, i) => `${i * 40 + 20},${120 - d.p * 1.0}`).join(' ')}
                    fill="none" stroke="#6366f1" strokeWidth="2" strokeLinejoin="round"
                  />
                )}
                {/* Dots */}
                {chartData.map((d, i) => (
                  <g key={i}>
                    <circle cx={i * 40 + 20} cy={120 - d.p} r="4"
                      fill={d.p >= 70 ? '#ef4444' : d.p >= 50 ? '#eab308' : '#22c55e'} />
                    <text x={i * 40 + 20} y="135" textAnchor="middle" fontSize="8" fill="#9ca3af">
                      {d.t.split(':').slice(0, 2).join(':')}
                    </text>
                  </g>
                ))}
              </svg>
              <p className="text-xs text-red-400 mt-1">— 70% threshold</p>
            </div>

            {/* Flags by class */}
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-5">
              <h3 className="text-sm font-semibold text-gray-600 mb-4">Flags by Category</h3>
              <div className="space-y-3">
                {Object.entries(flagsByClass).map(([name, count]) => {
                  const max = Math.max(...Object.values(flagsByClass))
                  return (
                    <div key={name} className="flex items-center gap-3">
                      <div className="w-36 text-xs text-gray-600 truncate">{name}</div>
                      <div className="flex-1 bg-gray-100 rounded-full h-2">
                        <div
                          className="bg-red-400 h-2 rounded-full"
                          style={{ width: `${(count / max) * 100}%` }}
                        />
                      </div>
                      <div className="text-xs text-gray-500 w-6 text-right">{count}</div>
                    </div>
                  )
                })}
              </div>
            </div>
          </div>
        )}

        {/* Flag timeline */}
        <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-6">
          <h3 className="text-lg font-semibold text-gray-700 mb-6">
            Flag Timeline
            <span className="text-sm font-normal text-gray-400 ml-2">({flags.length} flags)</span>
          </h3>
          <FlagTimeline
            flags={flags}
            isAdmin={true}
            onAcknowledge={handleAcknowledge}
          />
        </div>
      </div>
    </div>
  )
}
