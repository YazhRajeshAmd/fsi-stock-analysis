import { useState, useEffect, useLayoutEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import styles from './OnboardingTour.module.css'

const TOUR_KEY = 'fsi_tour_done'
const PAD = 10
const TOOLTIP_W = 300
const GAP = 16

const STEPS = [
  {
    target: '[data-tour="header"]',
    title: 'AMD FSI Stock Analysis',
    body: 'This demo runs on AMD Instinct MI300X hardware with the ROCm open-source software stack — delivering LLM-powered financial analysis at GPU speed.',
    position: 'bottom',
  },
  {
    target: '[data-tour="form"]',
    title: 'Configure Your Analysis',
    body: 'Enter one or more stock tickers, set a date range, and choose an investor profile to tailor the recommendations to your strategy.',
    position: 'right',
  },
  {
    target: '[data-tour="submit"]',
    title: 'Run the Model',
    body: 'Click Analyze to send your request to the MI300X. The model processes all tickers in parallel — results typically return in a few seconds.',
    position: 'bottom',
  },
  {
    target: '[data-tour="results"]',
    title: 'View Your Results',
    body: 'Results are organized across three tabs: Technical Analysis, Recommendations, and Stock Charts with SMA-20 overlay.',
    position: 'left',
  },
  {
    target: '[data-tour="cards"]',
    title: 'Explore the Tech Stack',
    body: 'Click any card to learn more about the hardware and software powering this demo — from MI300X specs to ROCm documentation.',
    position: 'top',
  },
]

// Use bottom/right anchoring where possible to avoid transform conflicts with Framer Motion's y animation
function getTooltipPos(rect, position) {
  const centreX = Math.max(12, Math.min(rect.left + rect.width / 2 - TOOLTIP_W / 2, window.innerWidth - TOOLTIP_W - 12))

  if (position === 'bottom') return { top: rect.bottom + GAP, left: centreX, width: TOOLTIP_W }
  if (position === 'top')    return { bottom: window.innerHeight - rect.top + GAP, left: centreX, width: TOOLTIP_W }
  if (position === 'right')  return { top: rect.top, left: rect.right + GAP, width: TOOLTIP_W }
  if (position === 'left')   return { top: rect.top, left: rect.left - GAP - TOOLTIP_W, width: TOOLTIP_W }
}

export default function OnboardingTour() {
  const [step, setStep]       = useState(0)
  const [rect, setRect]       = useState(null)
  const [visible, setVisible] = useState(() => !localStorage.getItem(TOUR_KEY))

  function dismiss() {
    localStorage.setItem(TOUR_KEY, '1')
    setRect(null)
    setVisible(false)
  }

  function restart() {
    setStep(0)
    setRect(null)
    setVisible(true)
  }

  function next() {
    if (step < STEPS.length - 1) setStep((s) => s + 1)
    else dismiss()
  }

  const measureTarget = useCallback(() => {
    const el = document.querySelector(STEPS[step].target)
    if (!el) return

    const r = el.getBoundingClientRect()
    const inView = r.top >= 0 && r.bottom <= window.innerHeight

    if (inView) {
      // Already visible — measure immediately, no scroll needed
      setRect(r)
    } else {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' })
      setTimeout(() => setRect(el.getBoundingClientRect()), 180)
    }
  }, [step])

  useLayoutEffect(() => {
    if (!visible) return
    measureTarget()
  }, [step, visible, measureTarget])

  useEffect(() => {
    if (!visible) return
    window.addEventListener('resize', measureTarget)
    return () => window.removeEventListener('resize', measureTarget)
  }, [visible, measureTarget])

  const showTour     = visible && rect
  const current      = STEPS[step]
  const tooltipPos   = rect ? getTooltipPos(rect, current.position) : null

  return (
    <div className={styles.root}>
      {/* Demo trigger */}
      {!visible && (
        <button className={styles.demoBtn} onClick={restart} title="Replay tour">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          Tour
        </button>
      )}

      {/* Overlay */}
      {showTour && (
        <div
          className={styles.overlay}
          onClick={dismiss}
          style={{
            '--sx': `${rect.left   - PAD}px`,
            '--sy': `${rect.top    - PAD}px`,
            '--sw': `${rect.width  + PAD * 2}px`,
            '--sh': `${rect.height + PAD * 2}px`,
          }}
        />
      )}

      {/* Spotlight — no key prop so it animates between positions instead of remounting */}
      {showTour && (
        <motion.div
          className={styles.spotlight}
          animate={{
            left:    rect.left   - PAD,
            top:     rect.top    - PAD,
            width:   rect.width  + PAD * 2,
            height:  rect.height + PAD * 2,
            opacity: 1,
          }}
          initial={{
            left:    rect.left   - PAD,
            top:     rect.top    - PAD,
            width:   rect.width  + PAD * 2,
            height:  rect.height + PAD * 2,
            opacity: 0,
          }}
          transition={{ duration: 0.22, ease: [0.25, 0.46, 0.45, 0.94] }}
        />
      )}

      {/* Tooltip — crossfades between steps */}
      <AnimatePresence mode="wait">
        {showTour && tooltipPos && (
          <motion.div
            key={step}
            className={styles.tooltip}
            style={tooltipPos}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{    opacity: 0, y: 6 }}
            transition={{ duration: 0.18, ease: 'easeOut' }}
          >
            <div className={styles.tooltipStep}>Step {step + 1} of {STEPS.length}</div>
            <div className={styles.tooltipTitle}>{current.title}</div>
            <p className={styles.tooltipBody}>{current.body}</p>
            <div className={styles.tooltipFooter}>
              <button className={styles.skipBtn} onClick={dismiss}>Skip</button>
              <button className={styles.nextBtn} onClick={next}>
                {step < STEPS.length - 1 ? 'Next →' : 'Get Started'}
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
