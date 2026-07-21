import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import styles from './AnalysisForm.module.css'
import DatePicker from './DatePicker'

const INVESTOR_TYPES = ['Conservative', 'Moderate', 'Aggressive', 'Day Trader']

const STEPS = [
  'Enter stock symbols (e.g., "AAPL, MSFT, GOOGL")',
  'Set your analysis date range',
  'Select your investor profile (Conservative, Moderate, Aggressive, Day Trader)',
  'Click "Analyze Stocks" to start GPU-accelerated analysis',
  'Review results across different analysis perspectives',
]

export default function AnalysisForm({ values, onChange, onSubmit, loading }) {
  const [helpOpen, setHelpOpen] = useState(false)
  const helpRef = useRef(null)

  useEffect(() => {
    if (!helpOpen) return
    function handleClick(e) {
      if (helpRef.current && !helpRef.current.contains(e.target)) setHelpOpen(false)
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [helpOpen])

  return (
    <motion.div
      className={styles.panel}
      data-tour="form"
      initial={{ opacity: 0, x: -32 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, ease: 'easeOut', delay: 0.1 }}
    >
      <div className={styles.panelHeader}>
        <h2 className={styles.panelTitle}>Analysis Configuration</h2>
        <div className={styles.helpWrap} ref={helpRef}>
          <button
            type="button"
            className={styles.helpBtn}
            onClick={() => setHelpOpen(o => !o)}
            aria-label="Example usage"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
              <circle cx="12" cy="17" r="0.5" fill="currentColor" />
            </svg>
          </button>
          <AnimatePresence>
            {helpOpen && (
              <motion.div
                className={styles.helpPopover}
                initial={{ opacity: 0, y: -6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                transition={{ duration: 0.18, ease: 'easeOut' }}
              >
                <p className={styles.helpTitle}>Example Usage</p>
                <ol className={styles.helpList}>
                  {STEPS.map((step, i) => (
                    <li key={i} className={styles.helpItem}>
                      <span className={styles.helpNum}>{i + 1}</span>
                      {step}
                    </li>
                  ))}
                </ol>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      <div className={styles.field}>
        <label className={styles.label}>Stock Symbol(s)</label>
        <input
          className={styles.input}
          type="text"
          placeholder="e.g. AAPL, MSFT, GOOGL"
          value={values.symbols}
          onChange={(e) => onChange('symbols', e.target.value)}
        />
      </div>

      <div className={styles.row}>
        <div className={styles.field}>
          <label className={styles.label}>Start Date</label>
          <DatePicker
            value={values.startDate}
            onChange={(v) => onChange('startDate', v)}
          />
        </div>
        <div className={styles.field}>
          <label className={styles.label}>End Date</label>
          <DatePicker
            value={values.endDate}
            onChange={(v) => onChange('endDate', v)}
          />
        </div>
      </div>

      <div className={styles.field}>
        <label className={styles.label}>Investor Type</label>
        <select
          className={styles.select}
          value={values.investorType}
          onChange={(e) => onChange('investorType', e.target.value)}
        >
          {INVESTOR_TYPES.map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
      </div>

      <motion.button
        className={styles.btn}
        data-tour="submit"
        onClick={onSubmit}
        disabled={loading}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        transition={{ type: 'spring', stiffness: 400, damping: 20 }}
      >
        {loading ? 'Analyzing...' : 'Analyze Stocks'}
      </motion.button>

      <p className={styles.disclaimer}>
        For educational purposes only. Not financial advice.
      </p>
    </motion.div>
  )
}
