import { Skeleton } from "@/components/ui/skeleton"
import type { CurrentAQIResponse } from "@/lib/api"
import {
  Activity,
  Wind,
  Sun,
  Flame,
  CloudFog,
  Factory,
} from "lucide-react"

interface Props {
  data: CurrentAQIResponse | null
  loading: boolean
}

interface PollutantConfig {
  key: keyof CurrentAQIResponse["pollutants"]
  name: string
  subscript: string
  fullName: string
  unit: string
  threshold: number
  whoLimit: string
  icon: React.ElementType
}

const POLLUTANTS: PollutantConfig[] = [
  { 
    key: "pm2_5", 
    name: "PM", 
    subscript: "2.5", 
    fullName: "Fine Particulate Matter", 
    unit: "µg/m³", 
    threshold: 35.4, 
    whoLimit: "15 µg/m³ (24h)",
    icon: Activity 
  },
  { 
    key: "pm10", 
    name: "PM", 
    subscript: "10", 
    fullName: "Coarse Inhalable Particles", 
    unit: "µg/m³", 
    threshold: 154, 
    whoLimit: "45 µg/m³ (24h)",
    icon: CloudFog 
  },
  { 
    key: "ozone", 
    name: "O", 
    subscript: "3", 
    fullName: "Ground-Level Ozone", 
    unit: "µg/m³", 
    threshold: 100, 
    whoLimit: "100 µg/m³ (8h)",
    icon: Sun 
  },
  { 
    key: "nitrogen_dioxide", 
    name: "NO", 
    subscript: "2", 
    fullName: "Nitrogen Dioxide", 
    unit: "µg/m³", 
    threshold: 100, 
    whoLimit: "25 µg/m³ (24h)",
    icon: Factory 
  },
  { 
    key: "sulphur_dioxide", 
    name: "SO", 
    subscript: "2", 
    fullName: "Sulphur Dioxide", 
    unit: "µg/m³", 
    threshold: 75, 
    whoLimit: "40 µg/m³ (24h)",
    icon: Flame 
  },
  { 
    key: "carbon_monoxide", 
    name: "CO", 
    subscript: "", 
    fullName: "Carbon Monoxide", 
    unit: "µg/m³", 
    threshold: 1000, 
    whoLimit: "4 mg/m³ (24h)",
    icon: Wind 
  },
]

export function PollutantCards({ data, loading }: Props) {
  if (loading || !data) {
    return (
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="glass-card rounded-xl p-4">
            <Skeleton className="h-4 w-16 mb-2" />
            <Skeleton className="h-8 w-20 mb-2" />
            <Skeleton className="h-2 w-full rounded-full" />
          </div>
        ))}
      </div>
    )
  }

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
      {POLLUTANTS.map((p) => {
        const Icon = p.icon
        const value = data.pollutants[p.key]
        const ratio = value != null ? value / p.threshold : 0

        // Determine status tag and color
        let statusText = "Safe"
        let statusColor = "#10B981"
        let statusBg = "rgba(16, 185, 129, 0.15)"

        if (ratio >= 1.5) {
          statusText = "Severe"
          statusColor = "#EF4444"
          statusBg = "rgba(239, 68, 68, 0.15)"
        } else if (ratio >= 1.0) {
          statusText = "Elevated"
          statusColor = "#F97316"
          statusBg = "rgba(249, 115, 22, 0.15)"
        } else if (ratio >= 0.5) {
          statusText = "Moderate"
          statusColor = "#F59E0B"
          statusBg = "rgba(245, 158, 11, 0.15)"
        }

        const isDominant = data.dominant_pollutant.toLowerCase().includes(p.key.toLowerCase().replace("_", ""))

        return (
          <div
            key={p.key}
            className={`glass-card group relative flex flex-col justify-between overflow-hidden rounded-xl p-4 ${
              isDominant ? "ring-1 ring-primary/40 border-primary/40" : ""
            }`}
          >
            {/* Header with Vector Icon & Subscript Formula */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5 font-medium text-foreground">
                <div 
                  className="flex h-6 w-6 items-center justify-center rounded-md"
                  style={{ backgroundColor: statusBg, color: statusColor }}
                >
                  <Icon className="h-3.5 w-3.5" />
                </div>
                <span className="text-sm font-bold font-heading">
                  {p.name}
                  {p.subscript && <sub className="text-[10px] bottom-0">{p.subscript}</sub>}
                </span>
              </div>
              <span 
                className="inline-flex items-center rounded-full px-1.5 py-0.2 text-[10px] font-semibold"
                style={{ backgroundColor: statusBg, color: statusColor }}
              >
                {statusText}
              </span>
            </div>

            {/* Numerical Value */}
            <div className="my-2.5">
              <div className="flex items-baseline gap-1">
                <span className="text-2xl font-bold tracking-tight text-foreground font-heading tabular-nums">
                  {value != null ? value.toFixed(1) : "—"}
                </span>
                <span className="text-[11px] text-muted-foreground">{p.unit}</span>
              </div>
              <p className="text-[10px] text-muted-foreground/80 truncate mt-0.5" title={p.fullName}>
                {p.fullName}
              </p>
            </div>

            {/* Progress Relative to Standard Threshold */}
            <div className="space-y-1">
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-secondary/80">
                <div
                  className="h-full rounded-full transition-all duration-700 ease-out"
                  style={{
                    width: `${Math.min(Math.max(ratio * 100, 4), 100)}%`,
                    backgroundColor: statusColor,
                    boxShadow: `0 0 6px ${statusColor}60`,
                  }}
                />
              </div>
              <div className="flex items-center justify-between text-[9px] text-muted-foreground/70">
                <span>0</span>
                <span>Ref {p.threshold}</span>
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}
