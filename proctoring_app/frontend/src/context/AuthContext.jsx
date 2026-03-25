/**
 * AuthContext — Global authentication state shared across all components.
 *
 * Provides: user, token, isAuthenticated, isAdmin,
 *           login(), signup(), logout()
 * Persists the JWT in localStorage so the session survives page refresh.
 */

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../services/api'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [token, setToken]   = useState(() => localStorage.getItem('token'))
  const [user, setUser]     = useState(() => {
    const raw = localStorage.getItem('user')
    try { return raw ? JSON.parse(raw) : null } catch { return null }
  })

  const isAuthenticated = !!token
  const isAdmin = user?.role === 'admin'

  // Keep axios default header in sync
  useEffect(() => {
    if (token) {
      api.defaults.headers.common['Authorization'] = `Bearer ${token}`
    } else {
      delete api.defaults.headers.common['Authorization']
    }
  }, [token])

  const _storeAuth = useCallback((tokenVal, userVal) => {
    localStorage.setItem('token', tokenVal)
    localStorage.setItem('user', JSON.stringify(userVal))
    setToken(tokenVal)
    setUser(userVal)
    api.defaults.headers.common['Authorization'] = `Bearer ${tokenVal}`
  }, [])

  const login = useCallback(async (email, password) => {
    const res = await api.post('/auth/login', { email, password })
    const { access_token, user_id, name, role } = res.data
    _storeAuth(access_token, { id: user_id, name, role, email })
    return res.data
  }, [_storeAuth])

  const signup = useCallback(async (email, name, password, role = 'student') => {
    const res = await api.post('/auth/signup', { email, name, password, role })
    const { access_token, user_id, role: r, name: n } = res.data
    _storeAuth(access_token, { id: user_id, name: n, role: r, email })
    return res.data
  }, [_storeAuth])

  const logout = useCallback(() => {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    setToken(null)
    setUser(null)
    delete api.defaults.headers.common['Authorization']
  }, [])

  return (
    <AuthContext.Provider value={{ user, token, isAuthenticated, isAdmin, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>')
  return ctx
}
