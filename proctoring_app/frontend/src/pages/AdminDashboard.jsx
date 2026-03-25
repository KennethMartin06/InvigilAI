/**
 * AdminDashboard — Live monitoring and historical session overview for admins.
 *
 * Sections: Stats overview, Live sessions grid, Recent flags table.
 * Auto-refreshes every 10 seconds.
 */
import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import Navbar from '../components/Navbar'
import StatsOverview from '../components/StatsOverview'
import LiveSessionCard from '../components/LiveSessionCard'
import { adminApi, flagApi } from '../services/api'
import { statusHex } from '../utils/featureHelpers'

export default function AdminDashboard() {
  const navigate = useNavigate()
  const [stats, setStats]           = useState(null)
  const [activeSessions, setActive] = useState([])
  const [allSessions, setAll]       = useState([])
  const [recentFlags, setRecent]    = useState([])
  const [statusFilter, setFilter]   = useState('')
  const [loading, setLoading]       = useState(true)

  const fetchData = useCallback(async () => {
    try {
      const [statsRes, activeRes, allRes] = await Promise.all([
        adminApi.stats(),
        adminApi.activeSessions(),
        adminApi.sessions(statusFilter ? { status: statusFilter } : {}),
      ])
      setStats(statsRes.data)
      setActive(activeRes.data)
      setAll(allRes.data)
    } catch (err) {
      console.error('Dashboard fetch failed:', err)
    } finally {
      setLoading(false)
    }
  }, [statusFilter])

  useEffect(() => {
    fetchData()
    const t = setInterval(fetchData, 10000)
    return () => clearInterval(t)
  }, [fetchData])

  // Build recent flags from all sessions
  const flagsFromSessions = allSessions
    .filter(s => s.total_flags > 0)
    .sort((a, b) => b.total_flags - a.total_flags)
    .slice(0, 20)

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />

      <div className="max-w-screen-xl mx-auto px-6 py-8 space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-gray-800">Admin Dashboard</h1>
          <button onClick={fetchData} className="text-sm text-indigo-600 hover:underline">↻ Refresh</button>
        </div>

        {/* Stats */}
        <StatsOverview stats={stats} />

        {/* Live Sessions */}
        <section>
          <h2 className="text-lg font-semibold text-gray-700 mb-4">
            Live Sessions
            {activeSessions.length > 0 && (
              <span className="ml-2 bg-green-100 text-green-700 text-xs px-2 py-0.5 rounded-full">
                {activeSessions.length} active
              </span>
            )}
          </h2>
          {activeSessions.length === 0 ? (
            <div className="text-gray-400 text-sm bg-white rounded-xl p-6 border border-gray-200 text-center">
              No active exam sessions right now.
            </div>
          ) : (
            <div className="grid sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {activeSessions.map(s => (
                <LiveSessionCard
                  key={s.id}
                  session={s}
                  onClick={() => navigate(`/admin/session/${s.id}`)}
                />
              ))}
            </div>
          )}
        </section>

        {/* All Sessions Table */}
        <section>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-700">All Sessions</h2>
            <select
              value={statusFilter}
              onChange={e => setFilter(e.target.value)}
              className="border border-gray-300 rounded-lg px-3 py-1.5 text-sm"
            >
              <option value="">All Status</option>
              <option value="active">Active</option>
              <option value="completed">Completed</option>
              <option value="terminated">Terminated</option>
            </select>
          </div>

          <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  {['Student', 'Exam', 'Started', 'Duration', 'Status', 'Flags', 'Peak Score', ''].map(h => (
                    <th key={h} className="text-left px-4 py-3 text-xs font-semibold text-gray-500 uppercase tracking-wide">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {allSessions.length === 0 && !loading && (
                  <tr><td colSpan={8} className="text-center py-8 text-gray-400">No sessions found.</td></tr>
                )}
                {allSessions.map(s => {
                  const duration = s.end_time
                    ? Math.round((new Date(s.end_time) - new Date(s.start_time)) / 60000)
                    : null
                  const colour = statusHex(s.max_cheating_score)
                  return (
                    <tr key={s.id} className="hover:bg-gray-50">
                      <td className="px-4 py-3 font-medium text-gray-800">{s.student_name || `#${s.user_id}`}</td>
                      <td className="px-4 py-3 text-gray-500 max-w-xs truncate">{s.exam_title}</td>
                      <td className="px-4 py-3 text-gray-400">{new Date(s.start_time).toLocaleString()}</td>
                      <td className="px-4 py-3 text-gray-400">{duration != null ? `${duration}m` : '—'}</td>
                      <td className="px-4 py-3">
                        <span className={`text-xs font-medium px-2 py-0.5 rounded-full
                          ${s.status === 'active' ? 'bg-green-100 text-green-700'
                          : s.status === 'completed' ? 'bg-blue-100 text-blue-700'
                          : 'bg-red-100 text-red-700'}`}>
                          {s.status}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-gray-700 font-medium">{s.total_flags}</td>
                      <td className="px-4 py-3 font-mono text-xs" style={{ color: colour }}>
                        {Math.round(s.max_cheating_score * 100)}%
                      </td>
                      <td className="px-4 py-3">
                        <button
                          onClick={() => navigate(`/admin/session/${s.id}`)}
                          className="text-xs text-indigo-600 hover:underline"
                        >
                          Review →
                        </button>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </section>

        {/* Flags by class distribution */}
        {stats?.flags_by_class && Object.keys(stats.flags_by_class).length > 0 && (
          <section>
            <h2 className="text-lg font-semibold text-gray-700 mb-4">Flags by Cheating Category</h2>
            <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
              <div className="space-y-2">
                {Object.entries(stats.flags_by_class).map(([name, count]) => {
                  const total = Object.values(stats.flags_by_class).reduce((a, b) => a + b, 0)
                  const pct   = total > 0 ? Math.round((count / total) * 100) : 0
                  return (
                    <div key={name} className="flex items-center gap-3">
                      <div className="w-36 text-sm text-gray-600 truncate">{name}</div>
                      <div className="flex-1 bg-gray-100 rounded-full h-2">
                        <div className="bg-indigo-500 h-2 rounded-full" style={{ width: `${pct}%` }} />
                      </div>
                      <div className="text-xs text-gray-500 w-16 text-right">{count} ({pct}%)</div>
                    </div>
                  )
                })}
              </div>
            </div>
          </section>
        )}
      </div>
    </div>
  )
}
