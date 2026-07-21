import { useState } from 'react'
import { Client } from '@gradio/client'
import Header from './components/Header'
import FeatureCards from './components/FeatureCards'
import AnalysisForm from './components/AnalysisForm'
import ResultsPanel from './components/ResultsPanel'
import { fetchChartData } from './lib/fetchChartData'
import { SAMPLE_RESULTS } from './lib/sampleData'
import Footer from './components/Footer'
import OnboardingTour from './components/OnboardingTour'

const DEFAULT_FORM = {
  symbols: '',
  startDate: '2024-01-01',
  endDate: '2024-12-31',
  investorType: 'Moderate',
}

export default function App() {
  const [form, setForm] = useState(DEFAULT_FORM)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)

  function handleChange(key, value) {
    setForm((prev) => ({ ...prev, [key]: value }))
  }

  async function handleSubmit() {
    if (!form.symbols.trim()) return
    setLoading(true)
    setError(null)
    setResults(null)

    try {
      const client = await Client.connect('http://localhost:7860')
      const response = await client.predict('/gradio_interface', {
        symbols:       form.symbols,
        start_date:    form.startDate,
        end_date:      form.endDate,
        investor_type: form.investorType,
      })

      // outputs: [ai_analysis, recommendations, chart, inference_time, token_count, data_points]
      const [aiAnalysis, recommendations, , inferenceTime, tokenCount, dataPoints] = response.data

      // Fetch chart data for the first symbol
      const firstSymbol = form.symbols.split(',')[0].trim().toUpperCase()
      let chartData = []
      try {
        chartData = await fetchChartData(firstSymbol, form.startDate, form.endDate)
      } catch (_) {
        // chart is non-critical — results still show without it
      }

      setResults({
        aiAnalysis,
        recommendations,
        chartData,
        chartSymbol: firstSymbol,
        metrics: { inferenceTime, tokenCount, dataPoints },
      })
    } catch (err) {
      setError(err.message ?? 'Analysis failed. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ position: 'relative' }}>
      <OnboardingTour />
      <Header />

      <div style={{
        maxWidth: '1100px',
        margin: '0 auto var(--space-xl)',
        padding: '0 var(--space-xl)',
        display: 'grid',
        gridTemplateColumns: '340px 1fr',
        gap: 'var(--space-md)',
        alignItems: 'stretch',
      }}>
        <AnalysisForm
          values={form}
          onChange={handleChange}
          onSubmit={handleSubmit}
          onPreview={() => setResults(SAMPLE_RESULTS)}
          loading={loading}
        />

        <ResultsPanel results={results} loading={loading} error={error} />
      </div>

      <div style={{ marginTop: 'var(--space-xl)' }}>
        <FeatureCards />
      </div>
      <Footer />
    </div>
  )
}
