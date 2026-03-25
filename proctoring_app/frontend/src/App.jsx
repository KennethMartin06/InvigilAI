/**
 * App.jsx — Root component: router setup with role-based protected routes.
 */
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider } from './context/AuthContext'
import ProtectedRoute from './components/ProtectedRoute'

import LoginPage      from './pages/LoginPage'
import SignupPage     from './pages/SignupPage'
import ExamLobby      from './pages/ExamLobby'
import ExamPage       from './pages/ExamPage'
import ExamComplete   from './pages/ExamComplete'
import AdminDashboard from './pages/AdminDashboard'
import SessionReview  from './pages/SessionReview'

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          {/* Public */}
          <Route path="/login"  element={<LoginPage />} />
          <Route path="/signup" element={<SignupPage />} />

          {/* Student */}
          <Route path="/lobby"   element={<ProtectedRoute><ExamLobby /></ProtectedRoute>} />
          <Route path="/exam/:sessionId" element={<ProtectedRoute><ExamPage /></ProtectedRoute>} />
          <Route path="/complete/:sessionId" element={<ProtectedRoute><ExamComplete /></ProtectedRoute>} />

          {/* Admin */}
          <Route path="/admin" element={<ProtectedRoute adminOnly><AdminDashboard /></ProtectedRoute>} />
          <Route path="/admin/session/:sessionId" element={<ProtectedRoute adminOnly><SessionReview /></ProtectedRoute>} />

          {/* Catch-all */}
          <Route path="*" element={<Navigate to="/login" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  )
}
