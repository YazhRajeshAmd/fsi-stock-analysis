import { motion } from 'framer-motion'
import styles from './FeatureCards.module.css'

const FEATURES = [
  {
    title: 'AMD Instinct MI300X',
    description: 'HBM3 memory bandwidth at 5.3 TB/s enables large-scale financial model inference at speed.',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <rect x="4" y="4" width="16" height="16" rx="2" />
        <rect x="9" y="9" width="6" height="6" />
        <line x1="9" y1="1" x2="9" y2="4" />
        <line x1="15" y1="1" x2="15" y2="4" />
        <line x1="9" y1="20" x2="9" y2="23" />
        <line x1="15" y1="20" x2="15" y2="23" />
        <line x1="20" y1="9" x2="23" y2="9" />
        <line x1="20" y1="14" x2="23" y2="14" />
        <line x1="1" y1="9" x2="4" y2="9" />
        <line x1="1" y1="14" x2="4" y2="14" />
      </svg>
    ),
  },
  {
    title: 'ROCm Software Stack',
    description: 'Open-source GPU compute stack delivering optimized LLM inference on AMD Instinct hardware.',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <polygon points="12 2 2 7 12 12 22 7 12 2" />
        <polyline points="2 17 12 22 22 17" />
        <polyline points="2 12 12 17 22 12" />
      </svg>
    ),
  },
  {
    title: 'Real-Time Processing',
    description: 'Technical indicators — RSI, SMA, MACD — computed on-GPU for sub-second market data processing.',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="13 2 13 9 19 9" />
        <path d="M19 2H13L3 12h7l-2 10 13-13h-7l5-9z" />
      </svg>
    ),
  },
  {
    title: 'Multi-Stock Analysis',
    description: 'Analyze multiple tickers in parallel — from single equities to full portfolio-level reporting.',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <line x1="18" y1="20" x2="18" y2="10" />
        <line x1="12" y1="20" x2="12" y2="4" />
        <line x1="6"  y1="20" x2="6"  y2="14" />
        <line x1="2"  y1="20" x2="22" y2="20" />
      </svg>
    ),
  },
]

const containerVariants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.1 } },
}

const cardVariants = {
  hidden:  { opacity: 0, y: 24 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: 'easeOut' } },
}

export default function FeatureCards() {
  return (
    <motion.section
      className={styles.section}
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {FEATURES.map((f) => (
        <motion.div key={f.title} className={styles.card} variants={cardVariants}>
          <div className={styles.header}>
            <span className={styles.accent} />
            <span className={styles.icon}>{f.icon}</span>
          </div>
          <h3 className={styles.title}>{f.title}</h3>
          <p className={styles.description}>{f.description}</p>
        </motion.div>
      ))}
    </motion.section>
  )
}
