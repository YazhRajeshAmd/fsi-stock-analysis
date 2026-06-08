import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import styles from './ResultsPanel.module.css'
import StockChart from './StockChart'
import AMDLoader from './AMDLoader'
import MetricsBar from './MetricsBar'

const TABS = [
  { id: 'analysis',        label: 'Technical Analysis' },
  { id: 'recommendations', label: 'Recommendations' },
  { id: 'charts',          label: 'Stock Charts' },
]

function EmptyState({ message }) {
  return (
    <div className={styles.empty}>
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--amd-border-mid)', marginBottom: 12 }}>
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
      </svg>
      <p>{message}</p>
    </div>
  )
}

function TextOutput({ text }) {
  if (!text) return <EmptyState message="Run an analysis to see results here." />
  return (
    <div className={styles.textOutput}>
      {text.split('\n').map((line, i) => (
        <p key={i} className={line.trim() === '' ? styles.spacer : undefined}>{line}</p>
      ))}
    </div>
  )
}

function ChartsTab({ chartData, chartSymbol }) {
  if (!chartData?.length) return <EmptyState message="Run an analysis to see charts here." />
  return <StockChart data={chartData} symbol={chartSymbol} />
}

export default function ResultsPanel({ results, loading, error }) {
  const [activeTab, setActiveTab] = useState('analysis')

  return (
    <motion.div
      className={styles.panel}
      data-tour="results"
      initial={{ opacity: 0, x: 32 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, ease: 'easeOut', delay: 0.2 }}
    >
      {/* Tab bar */}
      <div className={styles.tabBar}>
        {TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            className={[styles.tab, activeTab === tab.id ? styles.tabActive : ''].join(' ')}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
            {activeTab === tab.id && (
              <motion.span
                className={styles.tabIndicator}
                layoutId="tab-indicator"
                transition={{ type: 'spring', stiffness: 400, damping: 30 }}
              />
            )}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className={styles.content}>
        {error && (
          <div className={styles.error}>{error}</div>
        )}
        {loading ? (
          <AMDLoader />
        ) : (
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.2, ease: 'easeOut' }}
              className={styles.tabContent}
            >
              {activeTab === 'analysis' && (
                <TextOutput text={results?.aiAnalysis} />
              )}
              {activeTab === 'recommendations' && (
                <TextOutput text={results?.recommendations} />
              )}
              {activeTab === 'charts' && (
                <ChartsTab chartData={results?.chartData} chartSymbol={results?.chartSymbol} />
              )}
            </motion.div>
          </AnimatePresence>
        )}
      </div>

      <MetricsBar metrics={results?.metrics} />
    </motion.div>
  )
}
