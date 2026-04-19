/** LoginPage — Student / Admin login form with error handling. */
import { useState, useEffect } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function LoginPage() {
  const { login, isAuthenticated, isAdmin } = useAuth()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const [email, setEmail]       = useState('')
  const [password, setPassword] = useState('')
  const [error, setError]       = useState(null)
  const [loading, setLoading]   = useState(false)

  // Redirect if already logged in
  if (isAuthenticated) {
    navigate(isAdmin ? '/admin' : '/lobby', { replace: true })
  }

  // Auto-login when ?demo=1 is in the URL (used by portfolio link)
  useEffect(() => {
    if (searchParams.get('demo') === '1') {
      handleDemoLogin()
    }
  }, [])

  const handleDemoLogin = async () => {
    setError(null)
    setLoading(true)
    try {
      const data = await login('admin@proctor.ai', 'admin123')
      navigate(data.role === 'admin' ? '/admin' : '/lobby', { replace: true })
    } catch {
      setError('Demo account not ready — try manual login below.')
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      const data = await login(email, password)
      navigate(data.role === 'admin' ? '/admin' : '/lobby', { replace: true })
    } catch (err) {
      setError(err.response?.data?.detail || 'Login failed. Please check your credentials.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-white">InvigilAI</h1>
          <p className="text-gray-400 text-sm mt-1">AI-Powered Online Exam Proctoring</p>
        </div>

        <div className="bg-white rounded-2xl shadow-2xl p-8">
          <h2 className="text-xl font-semibold text-gray-800 mb-6">Sign In</h2>

          {error && (
            <div className="bg-red-50 border border-red-200 text-red-600 text-sm rounded-lg px-4 py-3 mb-5">
              {error}
            </div>
          )}

          {/* One-click Demo for portfolio visitors */}
          <button
            type="button"
            onClick={handleDemoLogin}
            disabled={loading}
            className="w-full mb-4 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold py-3 rounded-lg text-sm disabled:opacity-50 flex items-center justify-center gap-2"
          >
            {loading ? 'Loading demo…' : '▶  Try Live Demo — Admin Dashboard'}
          </button>

          <div className="flex items-center gap-3 mb-4">
            <div className="flex-1 h-px bg-gray-200" />
            <span className="text-xs text-gray-400">or sign in manually</span>
            <div className="flex-1 h-px bg-gray-200" />
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
              <input
                type="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                required
                placeholder="you@example.com"
                className="w-full border border-gray-300 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
              <input
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                required
                placeholder="••••••••"
                className="w-full border border-gray-300 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-2.5 rounded-lg text-sm disabled:opacity-50 mt-2"
            >
              {loading ? 'Signing in…' : 'Sign In'}
            </button>
          </form>

          <p className="text-center text-sm text-gray-500 mt-6">
            Don't have an account?{' '}
            <Link to="/signup" className="text-indigo-600 hover:underline font-medium">Sign Up</Link>
          </p>

          <div className="mt-4 pt-4 border-t border-gray-100 text-xs text-gray-400 text-center">
            Demo — Admin: admin@proctor.ai / admin123<br />
            Student: student1@proctor.ai / student123
          </div>
        </div>
      </div>
    </div>
  )
}
