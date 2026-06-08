import { motion } from 'framer-motion'
import styles from './Header.module.css'

const ROCM_LOGO = 'https://raw.githubusercontent.com/ROCm/rocm-docs-core/main/src/rocm_docs/rocm_docs_theme/static/images/rocm-logo.png'
const AMD_LOGO  = 'https://upload.wikimedia.org/wikipedia/commons/7/7c/AMD_Logo.svg'

export default function Header() {
  return (
    <motion.div
      className={styles.wrapper}
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
    >
      <div className={styles.pillGlow} data-tour="header">
      <header className={styles.header}>
        <div className={styles.inner}>

          {/* Left — AMD + separator + ROCm logo */}
          <div className={styles.logoLockup}>
            <img src={AMD_LOGO}  alt="AMD"  className={styles.amdLogo}  />
            <span className={styles.logoSeparator} />
            <img src={ROCM_LOGO} alt="ROCm" className={styles.rocmLogo} />
          </div>

          {/* Centre — nav labels */}
          <nav className={styles.nav}>
            <span className={styles.navItem}>Financial Analysis</span>
            <span className={styles.navDot} />
            <span className={styles.navItem}>ROCm Platform</span>
            <span className={styles.navDot} />
            <span className={styles.navItem}>AMD MI300X</span>
          </nav>

          {/* Right — live demo badge */}
          <div className={styles.badge}>
            <span className={styles.badgeDot} />
            Live Demo
          </div>

        </div>
      </header>
      </div>
    </motion.div>
  )
}
