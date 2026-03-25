/**
 * api.js — Axios instance pre-configured for the backend REST API.
 *
 * - baseURL points to /api (proxied by Vite to http://localhost:8000/api)
 * - Request interceptor attaches JWT from localStorage
 * - Response interceptor redirects to /login on 401
 */

import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 15000,
})

// Attach token on every request
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Handle auth errors globally
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token')
      localStorage.removeItem('user')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

// ── Typed API calls ─────────────────────────────────────────────────────────

export const authApi = {
  login:  (email, password)            => api.post('/auth/login',  { email, password }),
  signup: (email, name, password, role) => api.post('/auth/signup', { email, name, password, role }),
  me:     ()                           => api.get('/auth/me'),
}

export const examApi = {
  start:     (exam_title)                               => api.post('/exam/start', { exam_title }),
  answer:    (session_id, question_id, selected_answer) => api.post('/exam/answer', { session_id, question_id, selected_answer }),
  end:       (session_id)                               => api.post(`/exam/end/${session_id}`),
  questions: (exam_title)                               => api.get(`/exam/questions/${encodeURIComponent(exam_title)}`),
}

export const flagApi = {
  bySession: (session_id) => api.get(`/flags/${session_id}`),
  byStudent: (student_id) => api.get(`/flags/student/${student_id}`),
  acknowledge: (flag_id)  => api.patch(`/flags/${flag_id}/acknowledge`),
}

export const adminApi = {
  sessions:       (params) => api.get('/admin/sessions', { params }),
  activeSessions: ()       => api.get('/admin/sessions/active'),
  stats:          ()       => api.get('/admin/stats'),
  studentHistory: (id)     => api.get(`/admin/student/${id}/history`),
}

export default api
