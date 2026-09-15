import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import { motion } from 'framer-motion'
import styles from './StockChart.module.css'

function formatDate(dateStr) {
  const d = new Date(dateStr)
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

function formatPrice(v) {
  return `$${Number(v).toFixed(2)}`
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className={styles.tooltip}>
      <p className={styles.tooltipDate}>{label}</p>
      {payload.map((entry) => (
        <p key={entry.name} className={styles.tooltipRow} style={{ color: entry.color }}>
          {entry.name}: {formatPrice(entry.value)}
        </p>
      ))}
    </div>
  )
}

export default function StockChart({ data, symbol }) {
  if (!data?.length) return null

  return (
    <motion.div
      className={styles.wrapper}
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: 'easeOut' }}
    >
      <p className={styles.chartTitle}>{symbol} — Price History</p>
      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="#00C2DE" stopOpacity={0.15} />
              <stop offset="95%" stopColor="#00C2DE" stopOpacity={0}    />
            </linearGradient>
          </defs>

          <CartesianGrid stroke="#222426" strokeDasharray="3 3" vertical={false} />

          <XAxis
            dataKey="date"
            tickFormatter={formatDate}
            tick={{ fill: '#b4b9bc', fontSize: 11 }}
            axisLine={{ stroke: '#222426' }}
            tickLine={false}
            minTickGap={40}
          />
          <YAxis
            tickFormatter={(v) => `$${v}`}
            tick={{ fill: '#b4b9bc', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={58}
            domain={['auto', 'auto']}
          />

          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: 12, color: '#b4b9bc', paddingTop: 8 }}
          />

          {/* High/Low band */}
          <Area
            type="monotone"
            dataKey="high"
            stroke="none"
            fill="url(#areaGrad)"
            legendType="none"
            isAnimationActive={true}
            animationDuration={1000}
          />

          {/* Close price — teal */}
          <Line
            type="monotone"
            dataKey="close"
            name="Close"
            stroke="#00C2DE"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: '#00C2DE' }}
            isAnimationActive={true}
            animationDuration={1200}
            animationEasing="ease-out"
          />

          {/* SMA 20 — gold */}
          <Line
            type="monotone"
            dataKey="sma20"
            name="SMA 20"
            stroke="#C1A968"
            strokeWidth={1.5}
            dot={false}
            strokeDasharray="4 3"
            isAnimationActive={true}
            animationDuration={1400}
            animationEasing="ease-out"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </motion.div>
  )
}
