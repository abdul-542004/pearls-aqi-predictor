import { useEffect, useState, useMemo } from "react"
import { api } from "@/lib/api"
import type {
  CurrentAQIResponse,
  ForecastResponse,
  HistoryResponse,
  AnalyticsResponse,
} from "@/lib/api"
import { CurrentAQI } from "@/components/CurrentAQI"
import { DailyForecastCards } from "@/components/DailyForecastCards"
import { ForecastChart } from "@/components/ForecastChart"
import { HistoryChart } from "@/components/HistoryChart"
import { PollutantCards } from "@/components/PollutantCards"
import { Analytics } from "@/components/Analytics"
import { 
  Activity, 
  RefreshCw, 
  MapPin, 
  Cpu, 
  Radio, 
  Clock, 
  TrendingUp, 
  Sparkles,
  AlertCircle
} from "lucide-react"

export default function App() {
  const [current, setCurrent] = useState<CurrentAQIResponse | null>(null)
  const [forecast, setForecast] = useState<ForecastResponse | null>(null)
  const [history, setHistory] = useState<HistoryResponse | null>(null)
  const [analytics, setAnalytics] = useState<AnalyticsResponse | null>(null)

  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastRefreshed, setLastRefreshed] = useState<Date | null>(null)

  async function fetchAll() {
    setLoading(true)
    setError(null)
    try {
      const [c, f, h, a] = await Promise.all([
        api.getCurrentAQI(),
        api.getForecast(),
        api.getHistory(),
        api.getAnalytics(),
      ])
      setCurrent(c)
      setForecast(f)
      setHistory(h)
      setAnalytics(a)
      setLastRefreshed(new Date())
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to connect to backend service")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchAll()
  }, [])

  // Dynamic ambient glow color based on live AQI
  const ambientGlowColor = useMemo(() => {
    if (!current?.category?.color) return "oklch(0.22 0.06 220)"
    return current.category.color
  }, [current])

  return (
    <div className="relative min-h-screen overflow-x-hidden bg-background text-foreground selection:bg-primary/30 selection:text-foreground">
      {/* Ambient Atmospheric Backdrop Glow */}
      <div 
        className="pointer-events-none fixed inset-0 z-0 transition-opacity duration-1000"
        style={{
          background: `radial-gradient(ellipse 90% 40% at 50% 0%, ${ambientGlowColor}18 0%, transparent 65%)`
        }}
      />

      {/* Top Telemetry Header */}
      <header className="sticky top-0 z-40 border-b border-border/80 bg-background/80 backdrop-blur-xl transition-all">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6">
          {/* Brand & Identity */}
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 text-primary border border-primary/20 shadow-xs">
              <Activity className="h-5 w-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-bold tracking-tight text-foreground sm:text-lg font-heading">
                  Pearls AQI
                </h1>
                <span className="hidden items-center gap-1 rounded-full bg-primary/10 px-2 py-0.5 text-[11px] font-semibold text-primary border border-primary/20 sm:inline-flex">
                  <Cpu className="h-3 w-3" />
                  ML v2.0
                </span>
              </div>
              <p className="hidden text-xs text-muted-foreground sm:block">
                Precision Atmospheric Intelligence & Multi-Horizon Forecasting
              </p>
            </div>
          </div>

          {/* Right Status Bar & Controls */}
          <div className="flex items-center gap-2 sm:gap-3">
            {/* Location Pill */}
            <div className="hidden items-center gap-1.5 rounded-lg border border-border/60 bg-muted/40 px-2.5 py-1 text-xs text-muted-foreground md:flex">
              <MapPin className="h-3.5 w-3.5 text-primary" />
              <span className="font-medium text-foreground">Karachi</span>
              <span className="text-[10px] text-muted-foreground/70 font-mono">24.86°N, 67.01°E</span>
            </div>

            {/* Live Inference Indicator */}
            <div className="flex items-center gap-1.5 rounded-lg border border-emerald-500/20 bg-emerald-500/10 px-2.5 py-1 text-xs font-medium text-emerald-400">
              <span className="relative flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500"></span>
              </span>
              <span className="text-[11px] uppercase tracking-wider font-semibold">Live</span>
            </div>

            {/* Refresh Button */}
            <button
              onClick={fetchAll}
              disabled={loading}
              className="inline-flex items-center gap-2 rounded-lg border border-border/80 bg-secondary/80 px-3 py-1.5 text-xs font-semibold text-foreground shadow-xs transition-all hover:bg-secondary hover:border-primary/40 disabled:opacity-50"
              title="Synchronize latest atmospheric observation"
            >
              <RefreshCw className={`h-3.5 w-3.5 text-primary ${loading ? "animate-spin" : ""}`} />
              <span className="hidden sm:inline">Refresh</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="relative z-10 mx-auto max-w-7xl space-y-8 px-4 py-6 sm:px-6 sm:py-8">
        {/* Error Alert */}
        {error && (
          <div className="flex items-center gap-3 rounded-xl border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive shadow-lg backdrop-blur-md">
            <AlertCircle className="h-5 w-5 shrink-0" />
            <div className="flex-1">
              <p className="font-semibold">Backend Connection Issue</p>
              <p className="text-xs text-destructive/80 mt-0.5">
                {error}. Please verify the FastAPI backend server is running on port 8000.
              </p>
            </div>
            <button
              onClick={fetchAll}
              className="rounded-lg bg-destructive/20 px-3 py-1 text-xs font-medium hover:bg-destructive/30"
            >
              Retry
            </button>
          </div>
        )}

        {/* 1. Live AQI Telemetry Hero Section */}
        <section className="animate-entrance">
          <CurrentAQI data={current} loading={loading} />
        </section>

        {/* 2. Pollutant Concentration Matrix */}
        <section className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Radio className="h-4 w-4 text-primary" />
              <h2 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground font-heading">
                Atmospheric Pollutant Matrix
              </h2>
            </div>
            <span className="text-[11px] text-muted-foreground/80">
              WHO & US-EPA Reference Baseline
            </span>
          </div>
          <PollutantCards data={current} loading={loading} />
        </section>

        {/* 3. 3-Day Forecast Outlook & Area Timeline */}
        <section className="space-y-4">
          <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <div className="flex items-center gap-2">
                <TrendingUp className="h-4 w-4 text-primary" />
                <h2 className="text-base font-bold tracking-tight text-foreground sm:text-lg font-heading">
                  Multi-Horizon Forecast Intelligence
                </h2>
              </div>
              <p className="text-xs text-muted-foreground mt-0.5">
                Direct XGBoost models trained on 1h–72h horizons integrating numerical weather predictions
              </p>
            </div>
            {forecast?.summary && (
              <div className="flex items-center gap-2 text-xs">
                <span className="rounded-md bg-secondary/80 border border-border/60 px-2.5 py-1 text-muted-foreground">
                  Peak: <strong className="text-foreground font-mono">{Math.round(forecast.summary.max_aqi)} AQI</strong>
                </span>
                <span className="rounded-md bg-secondary/80 border border-border/60 px-2.5 py-1 text-muted-foreground">
                  Avg: <strong className="text-foreground font-mono">{Math.round(forecast.summary.avg_aqi)} AQI</strong>
                </span>
              </div>
            )}
          </div>

          {/* Key Milestone Cards (+24h, +48h, +72h) */}
          <DailyForecastCards
            data={forecast}
            currentAqi={current?.us_aqi}
            loading={loading}
          />

          {/* High-Resolution Interactive Timeline Chart */}
          <ForecastChart data={forecast} loading={loading} />
        </section>

        {/* 4. Observed Historical Trends */}
        <section className="space-y-3">
          <div className="flex items-center gap-2">
            <Clock className="h-4 w-4 text-primary" />
            <h2 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground font-heading">
              48-Hour Historical Observation
            </h2>
          </div>
          <HistoryChart data={history} loading={loading} />
        </section>

        {/* 5. ML Architecture, SHAP & Explainability */}
        <section className="space-y-3">
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-primary" />
            <h2 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground font-heading">
              Model Intelligence & SHAP Explainability
            </h2>
          </div>
          <Analytics data={analytics} loading={loading} />
        </section>

        {/* System Footer */}
        <footer className="mt-12 rounded-2xl border border-border/60 bg-muted/20 p-6 text-center text-xs text-muted-foreground backdrop-blur-md">
          <div className="flex flex-col items-center justify-between gap-4 sm:flex-row">
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 rounded-full bg-primary" />
              <span className="font-semibold text-foreground">Pearls AQI Intelligence Platform</span>
            </div>
            <div className="flex flex-wrap items-center justify-center gap-x-4 gap-y-1 text-[11px]">
              <span>Direct Multi-Horizon XGBoost</span>
              <span>•</span>
              <span>Open-Meteo Atmosphere API</span>
              <span>•</span>
              <span>Hopsworks Feature Store</span>
            </div>
            {lastRefreshed && (
              <span className="text-[10px] text-muted-foreground/70 font-mono">
                Last synced {lastRefreshed.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
              </span>
            )}
          </div>
        </footer>
      </main>
    </div>
  )
}
