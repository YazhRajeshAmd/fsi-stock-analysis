import { useState, useEffect, useRef } from 'react'
import styles from './DatePicker.module.css'

const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

function isoFor(y, m, d) {
  return `${y}-${String(m + 1).padStart(2, '0')}-${String(d).padStart(2, '0')}`
}

function parseISO(iso) {
  const [y, m, d] = iso.split('-').map(Number)
  return { y, m: m - 1, d }
}

function formatDisplay(iso) {
  if (!iso) return ''
  const { y, m, d } = parseISO(iso)
  return new Date(y, m, d).toLocaleDateString('en-US', {
    month: 'short', day: 'numeric', year: 'numeric',
  })
}

export default function DatePicker({ value, onChange, placeholder = 'Select date' }) {
  const today = new Date().toISOString().split('T')[0]
  const initial = value ? parseISO(value) : parseISO(today)

  const [open, setOpen] = useState(false)
  const [viewYear, setViewYear] = useState(initial.y)
  const [viewMonth, setViewMonth] = useState(initial.m)
  const ref = useRef(null)

  // Close on outside click
  useEffect(() => {
    if (!open) return
    function handleClick(e) {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false)
    }
    function handleKey(e) {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', handleClick)
    document.addEventListener('keydown', handleKey)
    return () => {
      document.removeEventListener('mousedown', handleClick)
      document.removeEventListener('keydown', handleKey)
    }
  }, [open])

  // Re-sync view when value changes externally
  useEffect(() => {
    if (value) {
      const { y, m } = parseISO(value)
      setViewYear(y)
      setViewMonth(m)
    }
  }, [value])

  function prevMonth() {
    if (viewMonth === 0) { setViewMonth(11); setViewYear(y => y - 1) }
    else setViewMonth(m => m - 1)
  }

  function nextMonth() {
    if (viewMonth === 11) { setViewMonth(0); setViewYear(y => y + 1) }
    else setViewMonth(m => m + 1)
  }

  function jumpToday() {
    const t = parseISO(today)
    setViewYear(t.y)
    setViewMonth(t.m)
  }

  function buildGrid() {
    const firstDow = new Date(viewYear, viewMonth, 1).getDay()
    const leading = firstDow === 0 ? 6 : firstDow - 1
    const daysInMonth = new Date(viewYear, viewMonth + 1, 0).getDate()
    const prevLast = new Date(viewYear, viewMonth, 0).getDate()
    const cells = []

    for (let i = leading - 1; i >= 0; i--) {
      const py = viewMonth === 0 ? viewYear - 1 : viewYear
      const pm = viewMonth === 0 ? 11 : viewMonth - 1
      cells.push({ iso: isoFor(py, pm, prevLast - i), day: prevLast - i, other: true })
    }
    for (let d = 1; d <= daysInMonth; d++) {
      cells.push({ iso: isoFor(viewYear, viewMonth, d), day: d, other: false })
    }
    const trailing = 42 - cells.length
    for (let d = 1; d <= trailing; d++) {
      const ny = viewMonth === 11 ? viewYear + 1 : viewYear
      const nm = viewMonth === 11 ? 0 : viewMonth + 1
      cells.push({ iso: isoFor(ny, nm, d), day: d, other: true })
    }
    return cells
  }

  const monthLabel = new Date(viewYear, viewMonth, 1).toLocaleDateString('en-US', {
    month: 'long', year: 'numeric',
  })

  return (
    <div className={styles.root} ref={ref}>
      <button
        type="button"
        className={styles.trigger}
        onClick={() => setOpen(o => !o)}
        aria-label={value ? `Selected date: ${formatDisplay(value)}. Click to change` : placeholder}
        aria-expanded={open}
        aria-haspopup="dialog"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect x="3" y="4" width="18" height="18" rx="2" />
          <path d="M16 2v4M8 2v4M3 10h18" />
        </svg>
        <span className={value ? styles.triggerValue : styles.triggerPlaceholder}>
          {value ? formatDisplay(value) : placeholder}
        </span>
      </button>

      {open && (
        <div className={styles.popover} role="dialog" aria-label="Date picker" aria-modal="true">
          <div className={styles.header}>
            <button type="button" className={styles.navBtn} onClick={prevMonth} aria-label="Previous month">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="15 18 9 12 15 6" />
              </svg>
            </button>
            <button type="button" className={styles.monthLabel} onClick={jumpToday} title="Jump to today">
              {monthLabel}
            </button>
            <button type="button" className={styles.navBtn} onClick={nextMonth} aria-label="Next month">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="9 18 15 12 9 6" />
              </svg>
            </button>
          </div>

          <div className={styles.weekRow} role="row">
            {DAYS.map(d => <span key={d} className={styles.weekDay} aria-label={d}>{d}</span>)}
          </div>

          <div className={styles.grid} role="grid" aria-label={monthLabel}>
            {buildGrid().map(({ iso, day, other }) => {
              const isSelected = iso === value
              const isToday = iso === today
              return (
                <button
                  key={iso}
                  type="button"
                  role="gridcell"
                  aria-label={`${formatDisplay(iso)}${isToday ? ', today' : ''}${isSelected ? ', selected' : ''}`}
                  aria-selected={isSelected}
                  aria-disabled={other}
                  className={[
                    styles.cell,
                    other     ? styles.cellOther    : '',
                    isToday   ? styles.cellToday    : '',
                    isSelected && !isToday ? styles.cellSelected : '',
                  ].join(' ')}
                  onClick={() => { onChange(iso); setOpen(false) }}
                >
                  {day}
                  {isToday && !isSelected && <span className={styles.dot} />}
                </button>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
