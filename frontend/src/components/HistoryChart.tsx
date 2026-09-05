import { useState } from "react"
import { Skeleton } from "@/components/ui/skeleton"
import type { HistoryResponse } from "@/lib/api"
import {
  Line,
  LineChart,
  CartesianGrid,
  XAxis,
  YAxis,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
} from "recharts"
import { Clock } from "lucide-react"

interface Props {
  data: HistoryResponse | null
  loading: boolean
}

const AQI_BANDS = [
  { y: 50, label: "Good (50)", color: "#10B981" },
  { y: 100, label: "Moderate (100)", color: "#F59E0B" },
  { y: 150, label: "USG (150)", color: "#F97316" },
  { y: 200, label: "Unhealthy (200)", color: "#EF4444" },
]

export function HistoryChart({ data, loading }: Props) {
  const [viewMode, setViewMode] = useState<"both" | "aqi" | "pm25">("both")

  if (loading || !data) {
    return (
      <div className="glass-card rounded-2xl p-6">
        <Skeleton className="h-6 w-44 mb-2" />
        <Skeleton className="h-4 w-60 mb-4" />
        <Skeleton className="h-64 w-full rounded-xl" />
      </div>
    )
  }

  const chartData = data.history.map((p) => ({
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
    aqi: Math.round(p.us_aqi),
    pm25: p.pm2_5 != null ? Math.round(p.pm2_5 * 10) / 10 : null,
    temp: p.temperature_2m,
    category: p.category.level,
    color: p.category.color || "#10B981",
  }))

  return (
    <div className="glass-panel rounded-2xl p-5 sm:p-6">
      <div className="flex flex-col gap-4 border-b border-border/50 pb-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-primary/10 text-primary">
              <Clock className="h-4 w-4" />
            </div>
            <h3 className="text-base font-bold text-foreground font-heading">
              Observed Environmental Trajectory (Last {data.history_hours}h)
            </h3>
          </div>
          <p className="mt-0.5 text-xs text-muted-foreground">
            Station ground-truth measurements captured across diurnal cycle
          </p>
        </div>

        {/* View Mode Toggle */}
        <div className="flex items-center gap-1 rounded-xl bg-secondary/80 p-1 border border-border/60">
          <button
            onClick={() => setViewMode("both")}
            className={`rounded-lg px-2.5 py-1 text-xs font-semibold transition-all ${
              viewMode === "both"
                ? "bg-primary text-primary-foreground shadow-xs"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            All Metrics
          </button>
          <button
            onClick={() => setViewMode("aqi")}
            className={`rounded-lg px-2.5 py-1 text-xs font-semibold transition-all ${
              viewMode === "aqi"
                ? "bg-primary text-primary-foreground shadow-xs"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            AQI Only
          </button>
          <button
            onClick={() => setViewMode("pm25")}
            className={`rounded-lg px-2.5 py-1 text-xs font-semibold transition-all ${
              viewMode === "pm25"
                ? "bg-primary text-primary-foreground shadow-xs"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            PM2.5 Only
          </button>
        </div>
      </div>

      <div className="h-64 w-full pt-4">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.25 0.018 255 / 0.5)" vertical={false} />
            <XAxis
              dataKey="time"
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              stroke="oklch(0.65 0.02 255)"
              fontSize={11}
              interval={5}
              tickFormatter={(val: string) => {
                const d = new Date(val)
                const day = d.toLocaleDateString([], { weekday: "short" })
                const hour = d.toLocaleTimeString([], { hour: "numeric", hour12: true })
                return `${day} ${hour}`
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
              content={<CustomHistoryTooltip />} 
              cursor={{ stroke: "oklch(0.4 0.04 255 / 0.5)", strokeWidth: 1, strokeDasharray: "3 3" }} 
            />
            
            {AQI_BANDS.map((band) => (
              <ReferenceLine
                key={band.y}
                y={band.y}
                stroke={band.color}
                strokeDasharray="4 4"
                strokeOpacity={0.3}
              />
            ))}

            {(viewMode === "both" || viewMode === "aqi") && (
              <Line
                type="monotone"
                dataKey="aqi"
                name="US AQI"
                stroke="#10B981"
                strokeWidth={2.5}
                dot={false}
                activeDot={{ r: 4, fill: "#10B981" }}
              />
            )}

            {(viewMode === "both" || viewMode === "pm25") && (
              <Line
                type="monotone"
                dataKey="pm25"
                name="PM2.5 (µg/m³)"
                stroke="#F59E0B"
                strokeWidth={2}
                strokeDasharray="4 3"
                dot={false}
                activeDot={{ r: 4, fill: "#F59E0B" }}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Legend Strip */}
      <div className="mt-3 flex items-center justify-center gap-6 border-t border-border/40 pt-2 text-xs">
        <div className="flex items-center gap-2">
          <span className="h-2.5 w-5 rounded bg-emerald-500" />
          <span className="text-muted-foreground">Observed AQI Index</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="h-1 w-5 rounded bg-amber-500 border-dashed border" />
          <span className="text-muted-foreground">PM₂.₅ Concentration (µg/m³)</span>
        </div>
      </div>
    </div>
  )
}

function CustomHistoryTooltip({ active, payload }: any) {
  if (!active || !payload || !payload.length) return null
  const data = payload[0].payload

  return (
    <div className="glass-card rounded-xl border border-border/80 bg-popover/95 p-3.5 shadow-xl backdrop-blur-md">
      <div className="border-b border-border/60 pb-1.5">
        <span className="text-xs font-semibold text-foreground font-mono">
          {data.formattedTime}
        </span>
      </div>

      <div className="my-2 space-y-1 text-xs">
        <div className="flex items-center justify-between gap-4">
          <span className="text-muted-foreground">US AQI:</span>
          <span className="font-bold font-mono text-emerald-400">{data.aqi}</span>
        </div>
        {data.pm25 != null && (
          <div className="flex items-center justify-between gap-4">
            <span className="text-muted-foreground">PM₂.₅:</span>
            <span className="font-bold font-mono text-amber-400">{data.pm25} µg/m³</span>
          </div>
        )}
      </div>

      <div className="border-t border-border/40 pt-1.5 text-[10px] text-muted-foreground">
        Category: <strong className="text-foreground">{data.category}</strong>
      </div>
    </div>
  )
}
