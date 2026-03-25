/**
 * StatsOverview — Row of metric cards for the admin dashboard.
 *
 * Props: stats — AdminStatsResponse object
 */
export default function StatsOverview({ stats }) {
  if (!stats) return <div className="text-gray-400 text-sm">Loading stats…</div>

  const cards = [
    { label: 'Active Sessions',    value: stats.active_sessions,    colour: 'text-green-600',  bg: 'bg-green-50' },
    { label: 'Total Flags Today',  value: stats.total_flags_today,  colour: 'text-red-600',    bg: 'bg-red-50' },
    { label: 'Avg Cheating Score', value: `${Math.round(stats.avg_cheating_score * 100)}%`, colour: 'text-orange-600', bg: 'bg-orange-50' },
    { label: 'High-Risk Students', value: stats.high_risk_students, colour: 'text-purple-600', bg: 'bg-purple-50' },
    { label: 'Total Sessions',     value: stats.total_sessions,     colour: 'text-indigo-600', bg: 'bg-indigo-50' },
    { label: 'All-Time Flags',     value: stats.total_flags_all_time, colour: 'text-gray-700', bg: 'bg-gray-50' },
  ]

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
      {cards.map(c => (
        <div key={c.label} className={`${c.bg} rounded-xl p-4 border border-gray-100 shadow-sm`}>
          <div className={`text-2xl font-bold ${c.colour}`}>{c.value}</div>
          <div className="text-xs text-gray-500 mt-1">{c.label}</div>
        </div>
      ))}
    </div>
  )
}
