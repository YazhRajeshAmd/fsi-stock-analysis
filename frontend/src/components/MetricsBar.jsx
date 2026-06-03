import { motion } from 'framer-motion'
import { useCountUp } from '../lib/useCountUp'
import styles from './MetricsBar.module.css'

// Extracts the first number found in a string e.g. "[AAPL] Inference Time: 2.34" → 2.34
function parseNum(str) {
  if (str == null) return null
  const match = String(str).match(/[\d.]+/)
  return match ? Number(match[0]) : null
}

function MetricCard({ label, rawValue, unit, decimals, delay }) {
  const num = parseNum(rawValue)
  const count = useCountUp(num ?? 0, { duration: 1400, decimals })

  return (
    <motion.div
      className={styles.card}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: 'easeOut', delay }}
    >
      <span className={styles.value}>
        {num != null ? `${count}${unit}` : '—'}
      </span>
      <span className={styles.label}>{label}</span>
    </motion.div>
  )
}

export default function MetricsBar({ metrics }) {
  if (!metrics) return null

  return (
    <motion.div
      className={styles.bar}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      <MetricCard
        label="Inference Time"
        rawValue={metrics.inferenceTime}
        unit="s"
        decimals={2}
        delay={0}
      />
      <div className={styles.divider} />
      <MetricCard
        label="Token Count"
        rawValue={metrics.tokenCount}
        unit=""
        decimals={0}
        delay={0.1}
      />
      <div className={styles.divider} />
      <MetricCard
        label="Data Points"
        rawValue={metrics.dataPoints}
        unit=""
        decimals={0}
        delay={0.2}
      />
    </motion.div>
  )
}
