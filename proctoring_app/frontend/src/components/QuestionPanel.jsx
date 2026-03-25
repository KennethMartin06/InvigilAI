/**
 * QuestionPanel — Displays a single MCQ with navigation and auto-save.
 *
 * Props:
 *   questions  — array of question objects
 *   answers    — { [question_id]: selected_answer }
 *   onAnswer(questionId, answer)
 */
import { useState } from 'react'

const OPTION_LABELS = ['A', 'B', 'C', 'D']

export default function QuestionPanel({ questions = [], answers = {}, onAnswer }) {
  const [current, setCurrent] = useState(0)

  if (!questions.length) {
    return <div className="text-gray-500 text-center py-16">No questions loaded.</div>
  }

  const q = questions[current]
  const options = [q.option_a, q.option_b, q.option_c, q.option_d]
  const selected = answers[q.id]

  return (
    <div className="flex flex-col h-full">
      {/* Progress */}
      <div className="flex items-center justify-between mb-4">
        <span className="text-sm text-gray-500">
          Question <strong>{current + 1}</strong> of <strong>{questions.length}</strong>
        </span>
        <div className="flex gap-1">
          {questions.map((_, i) => (
            <button
              key={i}
              onClick={() => setCurrent(i)}
              className={`w-7 h-7 text-xs rounded ${
                i === current
                  ? 'bg-indigo-600 text-white'
                  : answers[questions[i].id]
                  ? 'bg-green-100 text-green-700 border border-green-300'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {i + 1}
            </button>
          ))}
        </div>
      </div>

      {/* Question text */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-5 flex-1">
        <p className="text-gray-800 font-medium text-base leading-relaxed">{q.question_text}</p>

        <div className="mt-5 space-y-3">
          {options.map((opt, i) => {
            const label = OPTION_LABELS[i]
            const isSelected = selected === label
            return (
              <button
                key={label}
                onClick={() => onAnswer?.(q.id, label)}
                className={`w-full flex items-center gap-3 p-3 rounded-lg border text-left transition
                  ${isSelected
                    ? 'border-indigo-500 bg-indigo-50 text-indigo-800'
                    : 'border-gray-200 hover:border-indigo-300 hover:bg-gray-50'}`}
              >
                <span className={`w-7 h-7 flex items-center justify-center rounded-full text-sm font-bold flex-shrink-0
                  ${isSelected ? 'bg-indigo-600 text-white' : 'bg-gray-100 text-gray-600'}`}>
                  {label}
                </span>
                <span className="text-sm">{opt}</span>
              </button>
            )
          })}
        </div>
      </div>

      {/* Navigation */}
      <div className="flex justify-between">
        <button
          onClick={() => setCurrent(c => Math.max(0, c - 1))}
          disabled={current === 0}
          className="px-4 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg disabled:opacity-40"
        >
          ← Previous
        </button>
        <button
          onClick={() => setCurrent(c => Math.min(questions.length - 1, c + 1))}
          disabled={current === questions.length - 1}
          className="px-4 py-2 text-sm bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg disabled:opacity-40"
        >
          Next →
        </button>
      </div>
    </div>
  )
}
