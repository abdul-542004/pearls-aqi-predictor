import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert"
import { Skeleton } from "@/components/ui/skeleton"
import type { CurrentAQIResponse } from "@/lib/api"
import {
  Thermometer,
  Droplets,
  Wind,
  Gauge,
  CloudRain,
  MapPin,
  TriangleAlert,
  HeartPulse,
  ShieldCheck,
  ShieldAlert,
  Info,
} from "lucide-react"

interface Props {
  data: CurrentAQIResponse | null
  loading: boolean
}

export function CurrentAQI({ data, loading }: Props) {
  if (loading || !data) {
    return (
      <div className="glass-card rounded-2xl p-6 sm:p-8">
        <div className="flex flex-col gap-6 lg:flex-row lg:items-center">
          <div className="flex flex-col items-center justify-center p-4 lg:w-1/3">
            <Skeleton className="h-36 w-36 rounded-full" />
            <Skeleton className="mt-4 h-6 w-28 rounded-full" />
          </div>
          <div className="flex-1 space-y-4">
            <Skeleton className="h-6 w-48" />
            <Skeleton className="h-16 w-full rounded-xl" />
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Skeleton className="h-20 rounded-xl" />
              <Skeleton className="h-20 rounded-xl" />
              <Skeleton className="h-20 rounded-xl" />
              <Skeleton className="h-20 rounded-xl" />
            </div>
          </div>
        </div>
      </div>
    )
  }

  const { us_aqi, category, dominant_pollutant, weather, is_hazardous, location, time } = data
  const aqiVal = Math.round(us_aqi)
  const aqiColor = category.color || "#10B981"
  const updatedAt = new Date(time)

  // Maximum scale benchmark for circular gauge
  const progressPercent = Math.min((aqiVal / 300) * 100, 100)
  const radius = 41
  const circumference = 2 * Math.PI * radius
  const strokeDashoffset = circumference - (circumference * progressPercent) / 100

  const healthBadges = getHealthBadges(aqiVal)

  return (
    <div className="space-y-4">
      {/* High Alert Banner for Hazardous Condition */}
      {is_hazardous && (
        <Alert variant="destructive" className="border-red-500/40 bg-red-950/40 backdrop-blur-md">
          <TriangleAlert className="h-4 w-4" />
          <AlertTitle className="font-heading tracking-wide">Hazardous Atmospheric Episode</AlertTitle>
          <AlertDescription className="text-xs">
            {category.advisory || "Air quality is at severe hazardous levels. Minimize all outdoor exposure immediately."}
          </AlertDescription>
        </Alert>
      )}

      {/* Main Glassmorphic Hero Telemetry Unit */}
      <div 
        className="glass-panel relative overflow-hidden rounded-2xl p-6 sm:p-8"
        style={{
          borderLeft: `4px solid ${aqiColor}`,
          background: `linear-gradient(135deg, ${aqiColor}12 0%, rgba(14, 20, 36, 0.95) 45%, rgba(8, 11, 18, 0.98) 100%)`
        }}
      >
        {/* Top Header Strip inside Card */}
        <div className="mb-6 flex flex-wrap items-center justify-between gap-2 border-b border-border/50 pb-4">
          <div className="flex items-center gap-2">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-primary/10 text-primary">
              <MapPin className="h-4 w-4" />
            </div>
            <div>
              <span className="text-sm font-semibold text-foreground font-heading">
                {location.name} Station
              </span>
              <span className="ml-2 text-xs text-muted-foreground/80">
                Lat {location.latitude.toFixed(2)}°, Lon {location.longitude.toFixed(2)}°
              </span>
            </div>
          </div>
          <span className="text-xs text-muted-foreground font-mono">
            Observed {updatedAt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
          </span>
        </div>

        {/* Hero Body Grid */}
        <div className="grid grid-cols-1 gap-8 lg:grid-cols-12 lg:items-center">
          {/* Radial Circular AQI Gauge (4 cols on lg) */}
          <div className="flex flex-col items-center justify-center lg:col-span-4">
            <div className="relative flex h-44 w-44 items-center justify-center">
              {/* SVG Circular Ring Gauge */}
              <svg className="h-full w-full -rotate-90 overflow-visible" viewBox="0 0 100 100">
                {/* Background Ring Track */}
                <circle
                  cx="50"
                  cy="50"
                  r={radius}
                  fill="none"
                  stroke="oklch(0.24 0.02 255 / 0.45)"
                  strokeWidth="3.5"
                />
                {/* Colored Progress Arc */}
                <circle
                  cx="50"
                  cy="50"
                  r={radius}
                  fill="none"
                  stroke={aqiColor}
                  strokeWidth="4"
                  strokeLinecap="round"
                  strokeDasharray={circumference}
                  strokeDashoffset={strokeDashoffset}
                  className="transition-all duration-1000 ease-out"
                />
              </svg>

              {/* Inner Numerical Telemetry */}
              <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                <span className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
                  US AQI
                </span>
                <span 
                  className="text-5xl font-black tracking-tight tabular-nums font-heading"
                  style={{ color: aqiColor }}
                >
                  {aqiVal}
                </span>
                <span className="text-[11px] font-medium text-muted-foreground/80 mt-0.5">
                  Index Point
                </span>
              </div>
            </div>

            {/* Severity Category Pill */}
            <div className="mt-4 flex flex-col items-center gap-1.5">
              <span
                className="inline-flex items-center rounded-full px-3.5 py-1 text-xs font-bold shadow-xs tracking-wide"
                style={{
                  backgroundColor: `${aqiColor}25`,
                  color: aqiColor,
                  border: `1px solid ${aqiColor}50`
                }}
              >
                {category.level}
              </span>
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Gauge className="h-3.5 w-3.5 text-primary" />
                <span>Primary Pollutant:</span>
                <span className="font-semibold text-foreground font-mono uppercase">
                  {dominant_pollutant.replace(/_/g, ".")}
                </span>
              </div>
            </div>
          </div>

          {/* Right Column: Advisory & Weather Matrix (8 cols on lg) */}
          <div className="space-y-5 lg:col-span-8">
            {/* Advisory Description Card */}
            <div className="rounded-xl border border-border/70 bg-secondary/40 p-4 backdrop-blur-sm">
              <div className="flex items-start gap-3">
                <div 
                  className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg"
                  style={{ backgroundColor: `${aqiColor}20`, color: aqiColor }}
                >
                  <HeartPulse className="h-4 w-4" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-foreground font-heading">
                    Health Guidance & Impact
                  </h3>
                  <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                    {category.advisory}
                  </p>
                </div>
              </div>

              {/* Sub-advisory pills */}
              <div className="mt-3 grid grid-cols-1 gap-2 pt-2 sm:grid-cols-2 border-t border-border/40">
                {healthBadges.map((badge, idx) => {
                  const Icon = badge.icon
                  return (
                    <div key={idx} className="flex items-center gap-2 text-[11px] text-muted-foreground">
                      <Icon className="h-3.5 w-3.5 text-primary shrink-0" />
                      <span className="truncate">{badge.desc}</span>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* Weather Telemetry Matrix Grid */}
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {/* Temperature */}
              <WeatherMetricCard
                icon={<Thermometer className="h-4 w-4 text-rose-400" />}
                label="Temperature"
                value={weather.temperature_2m != null ? `${weather.temperature_2m.toFixed(1)}°C` : "—"}
                subtext="2m Ambient"
              />

              {/* Relative Humidity */}
              <WeatherMetricCard
                icon={<Droplets className="h-4 w-4 text-cyan-400" />}
                label="Humidity"
                value={weather.relative_humidity_2m != null ? `${Math.round(weather.relative_humidity_2m)}%` : "—"}
                subtext={weather.relative_humidity_2m != null && weather.relative_humidity_2m > 70 ? "Humid Coastal" : "Normal"}
              />

              {/* Wind Speed & Direction */}
              <WeatherMetricCard
                icon={
                  <Wind className="h-4 w-4 text-sky-400" />
                }
                label="Wind Speed"
                value={weather.wind_speed_10m != null ? `${weather.wind_speed_10m.toFixed(1)} km/h` : "—"}
                subtext={
                  weather.wind_direction_10m != null 
                    ? `${getWindDirection(weather.wind_direction_10m)} (${Math.round(weather.wind_direction_10m)}°)` 
                    : "10m Vector"
                }
              />

              {/* Precipitation / Pressure */}
              <WeatherMetricCard
                icon={<CloudRain className="h-4 w-4 text-indigo-400" />}
                label="Precipitation"
                value={weather.precipitation != null ? `${weather.precipitation.toFixed(1)} mm` : "0.0 mm"}
                subtext={weather.pressure_msl != null ? `${Math.round(weather.pressure_msl)} hPa` : "MSL"}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function WeatherMetricCard({
  icon,
  label,
  value,
  subtext,
}: {
  icon: React.ReactNode
  label: string
  value: string
  subtext: string
}) {
  return (
    <div className="flex flex-col justify-between rounded-xl border border-border/60 bg-secondary/30 p-3 transition-colors hover:border-border">
      <div className="flex items-center justify-between text-muted-foreground">
        <span className="text-[11px] font-medium tracking-wide uppercase">{label}</span>
        {icon}
      </div>
      <div className="mt-2">
        <p className="text-lg font-bold tracking-tight text-foreground font-heading tabular-nums">
          {value}
        </p>
        <p className="text-[10px] text-muted-foreground/80 mt-0.5 truncate">
          {subtext}
        </p>
      </div>
    </div>
  )
}

function getWindDirection(deg: number): string {
  const directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
  const index = Math.round((deg % 360) / 22.5) % 16
  return directions[index]
}

function getHealthBadges(aqiVal: number) {
  if (aqiVal <= 50) {
    return [
      { label: "Air Quality: Ideal", icon: ShieldCheck, desc: "Safe for all outdoor recreation and ventilation." },
      { label: "Sensitive Groups", icon: HeartPulse, desc: "No special precautions needed." },
    ]
  } else if (aqiVal <= 100) {
    return [
      { label: "Air Quality: Acceptable", icon: Info, desc: "Moderate air quality. Unusually sensitive people should consider reducing prolonged outdoor exertion." },
      { label: "Sensitive Groups", icon: HeartPulse, desc: "Monitor for respiratory irritation." },
    ]
  } else if (aqiVal <= 150) {
    return [
      { label: "Sensitive Groups Notice", icon: TriangleAlert, desc: "Children, elderly & people with respiratory disease should limit outdoor exertion." },
      { label: "General Public", icon: ShieldAlert, desc: "Less likely to be affected, keep windows closed during rush hour." },
    ]
  } else {
    return [
      { label: "Health Alert: Elevated", icon: TriangleAlert, desc: "Active children, adults, and people with respiratory illness should avoid outdoor exertion." },
      { label: "Mask Recommended", icon: ShieldAlert, desc: "Wear particulate-filtering masks (N95/KN95) outdoors." },
    ]
  }
}

