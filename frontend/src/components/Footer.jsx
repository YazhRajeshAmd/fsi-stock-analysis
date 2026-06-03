import styles from './Footer.module.css'

export default function Footer() {
  const year = new Date().getFullYear()
  return (
    <footer className={styles.footer}>
      <div className={styles.inner}>
        <div className={styles.left}>
          <span className={styles.powered}>
            Powered by <span className={styles.accent}>AMD Instinct MI300X</span> &amp; <span className={styles.accent}>ROCm Open Software</span>
          </span>
        </div>
        <div className={styles.right}>
          <span className={styles.copy}>
            &copy; {year} Advanced Micro Devices, Inc. For demonstration purposes only.
          </span>
        </div>
      </div>
    </footer>
  )
}
