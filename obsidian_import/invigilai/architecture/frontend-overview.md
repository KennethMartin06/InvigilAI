---
title: Frontend Overview — InvigilAI
project: invigilai
tags: [frontend, react, deployment, architecture]
source: proctoring_app/frontend/
---

# Frontend Overview

React (Vite) SPA with two primary surfaces: the student exam UI and the admin proctoring dashboard.

## Pages

| Page | Role |
|------|------|
| `LoginPage.jsx` | JWT auth |
| `SignupPage.jsx` | User registration |
| `ExamLobby.jsx` | Pre-exam checks (camera/mic, system compatibility) |
| `ExamPage.jsx` | Active exam — question panel + webcam feed + live features streaming |
| `ExamComplete.jsx` | Post-exam confirmation |
| `AdminDashboard.jsx` | Live grid of active sessions with risk indicators |
| `SessionReview.jsx` | Post-session replay with Grad-CAM + SHAP overlays |

## Components

| Component | Purpose |
|-----------|---------|
| `CameraFeed.jsx` | WebRTC webcam capture + frame upload |
| `ExamTimer.jsx` | Countdown + auto-submit |
| `FlagTimeline.jsx` | Session timeline with suspicion markers |
| `LiveSessionCard.jsx` | Per-student card (grid item on admin dashboard) |
| `QuestionPanel.jsx` | Renders MCQ / short-answer questions |
| `StatsOverview.jsx` | Aggregate proctoring stats |
| `StatusIndicator.jsx` | Green/yellow/red risk badge |
| `WarningBanner.jsx` | Real-time student warnings (e.g. "please face the camera") |

## Key patterns

- **WebSocket for streaming** — bidirectional; server pushes risk updates
- **Protected routes** via `ProtectedRoute.jsx` wrapping auth-gated pages
- **Context for auth state** in `context/` (no redux)
- **Services layer** in `services/` abstracts API calls from components

## Admin dashboard surfaces

Per report section VI-H, the dashboard renders:
1. Active sessions grid with per-student risk colors
2. Temporal risk heatmap (per-student × time)
3. Live alert panel (chronological flags with modality icons)
4. Session replay with multi-modal evidence timeline
