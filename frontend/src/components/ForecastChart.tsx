import { useState } from "react"
import { Skeleton } from "@/components/ui/skeleton"
import type { ForecastResponse } from "@/lib/api"
import {
  Area,
  AreaChart,
  CartesianGrid,
  XAxis,
  YAxis,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
} from "recharts"
import { TrendingUp } from "lucide-react"

interface Props {
  data: ForecastResponse | null
  loading: boolean
}

const AQI_BANDS = [
  { y: 50, label: "Good (≤50)", color: "#10B981" },
  { y: 100, label: "Moderate (≤100)", color: "#F59E0B" },
  { y: 150, label: "USG (≤150)", color: "#F97316" },
  { y: 200, label: "Unhealthy (≤200)", color: "#EF4444" },
  { y: 300, label: "V. Unhealthy (≤300)", color: "#8B5CF6" },
]

export function ForecastChart({ data, loading }: Props) {
  const [horizonFilter, setHorizonFilter] = useState<72 | 48 | 24>(72)

  if (loading || !data) {
    return (
      <div className="glass-card rounded-2xl p-6">
        <div className="flex items-center justify-between mb-4">
          <Skeleton className="h-6 w-48" />
          <Skeleton className="h-8 w-32 rounded-lg" />
        </div>
        <Skeleton className="h-72 w-full rounded-xl" />
      </div>
    )
  }

  const { summary } = data
  const filteredPoints = data.forecast.filter((p) => p.hours_ahead <= horizonFilter)

  const chartData = filteredPoints.map((p) => ({
    time: p.time,
    formattedTime: new Date(p.time).toLocaleString([], {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }),
    shortTime: new Date(p.time).toLocaleTimeString([], {
      hour: "2-digit",
      hour12: true,
    }),
    aqi: Math.round(p.predicted_aqi),
    hours: p.hours_ahead,
    category: p.category.level,
    color: p.category.color || "#0284C7",
    temp: p.temperature_2m,
    humidity: p.relative_humidity_2m,
    wind: p.wind_speed_10m,
  }))

  return (
    <div className="glass-panel rounded-2xl p-5 sm:p-6">
      {/* Chart Header with Controls */}
      <div className="flex flex-col gap-4 border-b border-border/50 pb-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-primary/10 text-primary">
              <TrendingUp className="h-4 w-4" />
            </div>
            <h3 className="text-base font-bold text-foreground font-heading">
              72-Hour AQI Trend Simulation
            </h3>
          </div>
          <p className="mt-0.5 text-xs text-muted-foreground">
            Direct XGBoost Multi-Horizon trajectory with hourly meteorological integration
          </p>
        </div>

        {/* Time Window Segmented Control Buttons */}
        <div className="flex items-center gap-1 rounded-xl bg-secondary/80 p-1 border border-border/60">
          {( [24, 48, 72] as const ).map((hrs) => (
            <button
              key={hrs}
              onClick={() => setHorizonFilter(hrs)}
              className={`rounded-lg px-3 py-1 text-xs font-semibold transition-all ${
                horizonFilter === hrs
                  ? "bg-primary text-primary-foreground shadow-xs"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {hrs}h Window
            </button>
          ))}
        </div>
      </div>

      {/* Metric Summary Strip */}
      <div className="my-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <div className="rounded-xl border border-border/50 bg-secondary/30 p-2.5">
          <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Average Forecast</span>
          <p className="text-lg font-bold text-foreground font-mono tabular-nums">
            {Math.round(summary.avg_aqi)} <span className="text-xs text-muted-foreground font-normal">AQI</span>
          </p>
          <p className="text-[10px] text-muted-foreground/70 mt-0.5">Across {horizonFilter}h window</p>
        </div>

        <div className="rounded-xl border border-border/50 bg-secondary/30 p-2.5">
          <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Peak Pollution (Max)</span>
          <p className="text-lg font-bold text-rose-400 font-mono tabular-nums">
            {Math.round(summary.max_aqi)} <span className="text-xs text-muted-foreground font-normal">AQI</span>
          </p>
          <p className="text-[10px] text-rose-400/80 mt-0.5 truncate">
            {summary.peak_hour ? new Date(summary.peak_hour).toLocaleString([], { weekday: "short", hour: "numeric", hour12: true }) : "Peak hour"}
          </p>
        </div>

        <div className="rounded-xl border border-border/50 bg-secondary/30 p-2.5">
          <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Cleanest Window (Min)</span>
          <p className="text-lg font-bold text-emerald-400 font-mono tabular-nums">
            {Math.round(summary.min_aqi)} <span className="text-xs text-muted-foreground font-normal">AQI</span>
          </p>
          <p className="text-[10px] text-emerald-400/80 mt-0.5 truncate">
            {summary.cleanest_hour ? new Date(summary.cleanest_hour).toLocaleString([], { weekday: "short", hour: "numeric", hour12: true }) : "Cleanest hour"}
          </p>
        </div>

        <div className="rounded-xl border border-border/50 bg-secondary/30 p-2.5">
          <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Hazardous Hours</span>
          <p className="text-lg font-bold text-foreground font-mono tabular-nums">
            {summary.hazardous_hours_count} <span className="text-xs text-muted-foreground font-normal">hrs</span>
          </p>
          <p className="text-[10px] text-muted-foreground/70 mt-0.5">AQI &gt; 300 threshold</p>
        </div>
      </div>

      {/* Main Chart Graphic */}
      <div className="h-72 w-full pt-2">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="forecastAreaGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#0284C7" stopOpacity={0.4} />
                <stop offset="50%" stopColor="#0EA5E9" stopOpacity={0.15} />
                <stop offset="100%" stopColor="#0284C7" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.25 0.018 255 / 0.5)" vertical={false} />
            <XAxis
              dataKey="time"
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              stroke="oklch(0.65 0.02 255)"
              fontSize={11}
              interval={horizonFilter === 24 ? 3 : horizonFilter === 48 ? 6 : 9}
              tickFormatter={(val: string) => {
                const d = new Date(val)
                const day = d.toLocaleDateString([], { weekday: "short" })
                const hour = d.toLocaleTimeString([], { hour: "numeric", hour12: true })
                return horizonFilter === 24 ? hour : `${day} ${hour}`
              }}
            />
            <YAxis
              tickLine={false}
              axisLine={false}
              stroke="oklch(0.65 0.02 255)"
              fontSize={11}
              domain={[0, "auto"]}
            />
            <Tooltip 
              content={<CustomForecastTooltip />} 
              cursor={{ stroke: "oklch(0.4 0.04 255 / 0.5)", strokeWidth: 1, strokeDasharray: "3 3" }} 
            />
            {AQI_BANDS.map((band) => (
              <ReferenceLine
                key={band.y}
                y={band.y}
                stroke={band.color}
                strokeDasharray="4 4"
                strokeOpacity={0.4}
              />
            ))}
            <Area
              type="monotone"
              dataKey="aqi"
              name="Predicted AQI"
              stroke="#38BDF8"
              strokeWidth={2.5}
              fill="url(#forecastAreaGradient)"
              dot={false}
              activeDot={{ r: 5, fill: "#38BDF8", stroke: "#0B0F19", strokeWidth: 2 }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* EPA Severity Band Legend */}
      <div className="mt-4 flex flex-wrap items-center justify-center gap-x-4 gap-y-2 border-t border-border/40 pt-3">
        {AQI_BANDS.map((band) => (
          <div key={band.y} className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full" style={{ backgroundColor: band.color }} />
            <span className="text-[11px] text-muted-foreground">{band.label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function CustomForecastTooltip({ active, payload }: any) {
  if (!active || !payload || !payload.length) return null
  const data = payload[0].payload

  return (
    <div className="glass-card rounded-xl border border-border/80 bg-popover/95 p-3.5 shadow-xl backdrop-blur-md">
      <div className="flex items-center justify-between gap-3 border-b border-border/60 pb-2">
        <span className="text-xs font-semibold text-foreground font-mono">
          {data.formattedTime}
        </span>
        <span className="text-[10px] rounded bg-secondary px-1.5 py-0.5 text-muted-foreground font-mono">
          +{data.hours}h
        </span>
      </div>

      <div className="my-2 flex items-center justify-between gap-4">
        <div>
          <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Predicted AQI</span>
          <p className="text-2xl font-black font-heading tabular-nums" style={{ color: data.color }}>
            {data.aqi}
          </p>
        </div>
        <span
          className="rounded-full px-2.5 py-0.5 text-[11px] font-semibold"
          style={{ backgroundColor: `${data.color}20`, color: data.color, border: `1px solid ${data.color}40` }}
        >
          {data.category}
        </span>
      </div>

      {/* Atmospheric metrics at this point */}
      <div className="grid grid-cols-3 gap-2 border-t border-border/40 pt-2 text-[10px] text-muted-foreground font-mono">
        <div>Temp: <strong className="text-foreground">{data.temp != null ? `${data.temp.toFixed(1)}°C` : "—"}</strong></div>
        <div>Humidity: <strong className="text-foreground">{data.humidity != null ? `${Math.round(data.humidity)}%` : "—"}</strong></div>
        <div>Wind: <strong className="text-foreground">{data.wind != null ? `${data.wind.toFixed(0)}kph` : "—"}</strong></div>
      </div>
    </div>
  )
}
