import { Skeleton } from "@/components/ui/skeleton"
import type { ForecastResponse } from "@/lib/api"
import { 
  TrendingUp, 
  TrendingDown, 
  Minus, 
  Calendar, 
  Thermometer, 
  Droplets, 
  Wind,
} from "lucide-react"

interface Props {
  data: ForecastResponse | null
  currentAqi?: number | null
  loading: boolean
}

const HORIZONS = [
  { hours: 24, label: "+24 Hours", tag: "Day 1 Outlook" },
  { hours: 48, label: "+48 Hours", tag: "Day 2 Outlook" },
  { hours: 72, label: "+72 Hours", tag: "Day 3 Outlook" },
]

export function DailyForecastCards({ data, currentAqi, loading }: Props) {
  if (loading || !data) {
    return (
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        {HORIZONS.map((h) => (
          <div key={h.hours} className="glass-card rounded-2xl p-5">
            <Skeleton className="h-4 w-28 mb-3" />
            <Skeleton className="h-12 w-24 mb-3" />
            <Skeleton className="h-6 w-32 rounded-full mb-4" />
            <Skeleton className="h-10 w-full rounded-lg" />
          </div>
        ))}
      </div>
    )
  }

  // Map horizons to exact or closest available points
  const points = HORIZONS.map((h) => {
    const exact = data.forecast.find((p) => p.hours_ahead === h.hours)
    if (exact) return { ...h, point: exact }

    const sortedByDiff = [...data.forecast].sort(
      (a, b) => Math.abs(a.hours_ahead - h.hours) - Math.abs(b.hours_ahead - h.hours)
    )
    return { ...h, point: sortedByDiff[0] }
  })

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
      {points.map(({ hours, label, tag, point }) => {
        if (!point) return null

        const predictedAqi = Math.round(point.predicted_aqi)
        const aqiColor = point.category.color || "#F59E0B"
        const categoryLevel = point.category.level || "Moderate"

        // Calculate delta vs. current AQI
        let deltaText = ""
        let deltaType: "worsening" | "improving" | "steady" = "steady"
        let deltaValue = 0

        if (currentAqi != null) {
          deltaValue = point.predicted_aqi - currentAqi
          const absDelta = Math.abs(deltaValue).toFixed(1)

          if (deltaValue > 1.0) {
            deltaType = "worsening"
            deltaText = `+${absDelta} AQI Worsening`
          } else if (deltaValue < -1.0) {
            deltaType = "improving"
            deltaText = `-${absDelta} AQI Improving`
          } else {
            deltaType = "steady"
            deltaText = `Steady (±0)`
          }
        }

        const targetDate = new Date(point.time)
        const dateFormatted = targetDate.toLocaleDateString([], {
          weekday: "short",
          month: "short",
          day: "numeric",
        })
        const timeFormatted = targetDate.toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
        })

        return (
          <div
            key={hours}
            className="glass-card relative flex flex-col justify-between overflow-hidden rounded-2xl p-5"
            style={{
              borderTop: `3px solid ${aqiColor}`,
              background: `linear-gradient(180deg, ${aqiColor}10 0%, rgba(14, 20, 36, 0.9) 35%, rgba(8, 11, 18, 0.95) 100%)`
            }}
          >
            {/* Card Header: Horizon Badge & Target Time */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <span className="text-xs font-bold uppercase tracking-wider text-primary font-heading">
                  {label}
                </span>
                <span className="text-[10px] rounded bg-secondary/80 px-1.5 py-0.5 text-muted-foreground font-medium">
                  {tag}
                </span>
              </div>
              <span className="flex items-center gap-1 text-[11px] font-mono text-muted-foreground">
                <Calendar className="h-3 w-3 text-muted-foreground/70" />
                {dateFormatted}, {timeFormatted}
              </span>
            </div>

            {/* Middle: Predicted Index Point & Status Pill */}
            <div className="my-4 flex items-end justify-between">
              <div>
                <div className="flex items-baseline gap-1">
                  <span 
                    className="text-4xl font-extrabold tracking-tight tabular-nums font-heading"
                    style={{ color: aqiColor }}
                  >
                    {predictedAqi}
                  </span>
                  <span className="text-xs text-muted-foreground">AQI</span>
                </div>

                <div className="mt-1.5">
                  <span
                    className="inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold shadow-xs"
                    style={{
                      backgroundColor: `${aqiColor}20`,
                      color: aqiColor,
                      border: `1px solid ${aqiColor}40`
                    }}
                  >
                    {categoryLevel}
                  </span>
                </div>
              </div>

              {/* Delta Comparison Badge vs Current */}
              {deltaText && (
                <div className="text-right">
                  <span
                    className={`inline-flex items-center gap-1 rounded-lg px-2 py-1 text-xs font-semibold ${
                      deltaType === "worsening"
                        ? "bg-rose-500/10 text-rose-400 border border-rose-500/20"
                        : deltaType === "improving"
                        ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                        : "bg-muted text-muted-foreground border border-border"
                    }`}
                  >
                    {deltaType === "worsening" && <TrendingUp className="h-3 w-3 shrink-0" />}
                    {deltaType === "improving" && <TrendingDown className="h-3 w-3 shrink-0" />}
                    {deltaType === "steady" && <Minus className="h-3 w-3 shrink-0" />}
                    <span className="font-mono text-[11px]">{deltaText}</span>
                  </span>
                  <p className="mt-0.5 text-[9px] text-muted-foreground/70">vs. current baseline</p>
                </div>
              )}
            </div>

            {/* Meteorological Sub-Telemetry for this horizon */}
            <div className="grid grid-cols-3 gap-2 border-t border-border/50 pt-3 text-[11px]">
              <div className="flex items-center gap-1 text-muted-foreground">
                <Thermometer className="h-3 w-3 text-rose-400 shrink-0" />
                <span className="font-mono text-foreground">{point.temperature_2m != null ? `${point.temperature_2m.toFixed(0)}°C` : "—"}</span>
              </div>
              <div className="flex items-center gap-1 text-muted-foreground">
                <Droplets className="h-3 w-3 text-cyan-400 shrink-0" />
                <span className="font-mono text-foreground">{point.relative_humidity_2m != null ? `${Math.round(point.relative_humidity_2m)}%` : "—"}</span>
              </div>
              <div className="flex items-center gap-1 text-muted-foreground">
                <Wind className="h-3 w-3 text-sky-400 shrink-0" />
                <span className="font-mono text-foreground">{point.wind_speed_10m != null ? `${point.wind_speed_10m.toFixed(0)}kph` : "—"}</span>
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}
