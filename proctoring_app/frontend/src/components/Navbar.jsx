/** Navbar — Top navigation bar with user info and logout button. */
import { Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function Navbar() {
  const { user, isAdmin, logout } = useAuth()

  return (
    <nav className="bg-gray-900 text-white px-6 py-3 flex items-center justify-between shadow-lg">
      <Link to={isAdmin ? '/admin' : '/lobby'} className="flex items-center gap-2 font-bold text-lg">
        <span className="text-indigo-400">InvigilAI</span>
        <span className="text-xs bg-indigo-600 px-2 py-0.5 rounded-full uppercase tracking-wide">
          {isAdmin ? 'Admin' : 'Student'}
        </span>
      </Link>

      <div className="flex items-center gap-4 text-sm">
        <span className="text-gray-300">{user?.name}</span>
        <button
          onClick={logout}
          className="bg-gray-700 hover:bg-gray-600 px-3 py-1.5 rounded text-white"
        >
          Sign Out
        </button>
      </div>
    </nav>
  )
}
