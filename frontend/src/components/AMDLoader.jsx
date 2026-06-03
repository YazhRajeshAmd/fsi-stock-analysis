import { motion } from 'framer-motion'
import styles from './AMDLoader.module.css'

// AMD arrowmark — three chevrons with SVG shimmer scan across the shapes
function ArrowMark({ className }) {
  return (
    <svg
      className={className}
      viewBox="0 0 120 72"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        {/* Clip to arrow shapes so shimmer is masked to them */}
        <clipPath id="arrowClip">
          <polygon points="0,72 20,72 40,36 20,0 0,0 20,36" />
          <polygon points="40,72 60,72 80,36 60,0 40,0 60,36" />
          <polygon points="80,72 100,72 120,36 100,0 80,0 100,36" />
        </clipPath>
        {/* Shimmer gradient — bright streak from left to right */}
        <linearGradient id="shimmer" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%"   stopColor="#00C2DE" stopOpacity="0.5" />
          <stop offset="40%"  stopColor="#00C2DE" stopOpacity="0.5" />
          <stop offset="50%"  stopColor="#C1A968" stopOpacity="1" />
          <stop offset="60%"  stopColor="#00C2DE" stopOpacity="0.5" />
          <stop offset="100%" stopColor="#00C2DE" stopOpacity="0.5" />
          <animateTransform
            attributeName="gradientTransform"
            type="translate"
            from="-1 0"
            to="1 0"
            dur="1.6s"
            repeatCount="indefinite"
            additive="sum"
          />
        </linearGradient>
      </defs>

      {/* Base arrows at full teal */}
      <g opacity="0.4">
        <polygon points="0,72 20,72 40,36 20,0 0,0 20,36"       fill="#00C2DE" />
        <polygon points="40,72 60,72 80,36 60,0 40,0 60,36"     fill="#00C2DE" />
        <polygon points="80,72 100,72 120,36 100,0 80,0 100,36" fill="#00C2DE" />
      </g>

      {/* Shimmer layer — clipped to arrow shapes */}
      <rect
        x="0" y="0" width="120" height="72"
        fill="url(#shimmer)"
        clipPath="url(#arrowClip)"
      />
    </svg>
  )
}

export default function AMDLoader({ label = 'Analyzing with AMD MI300X...' }) {
  return (
    <div className={styles.wrapper}>
      {/* Arrowmark with built-in shimmer */}
      <div className={styles.iconWrap}>
        <ArrowMark className={styles.arrow} />
      </div>

      <p className={styles.label}>{label}</p>

      {/* Dot trail */}
      <div className={styles.dots}>
        {[0, 1, 2].map((i) => (
          <motion.span
            key={i}
            className={styles.dot}
            animate={{ opacity: [0.2, 1, 0.2] }}
            transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.2, ease: 'easeInOut' }}
          />
        ))}
      </div>
    </div>
  )
}
