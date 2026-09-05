import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import type { AnalyticsResponse } from "@/lib/api"
import {
  Bar,
  BarChart,
  CartesianGrid,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
  Cell,
} from "recharts"
import { 
  BrainCircuit, 
  BarChart3, 
  Sparkles, 
  Cpu, 
  Layers, 
  Wind, 
  Sun, 
  Flame,
} from "lucide-react"

interface Props {
  data: AnalyticsResponse | null
  loading: boolean
}

export function Analytics({ data, loading }: Props) {
  if (loading || !data) {
    return (
      <div className="glass-card rounded-2xl p-6">
        <Skeleton className="h-6 w-56 mb-4" />
        <Skeleton className="h-80 w-full rounded-xl" />
      </div>
    )
  }

  const report = data.training_report
  const shap1h = data.shap_importance["1h_model"] || []
  const shap24h = data.shap_importance["24h_model"] || []
  const horizons = report.results_per_horizon || {}

  const shapData1h = shap1h.slice(0, 12).map((f) => ({
    name: formatFeatureName(f.feature),
    rawName: f.feature,
    value: Math.round(f.mean_abs_shap * 1000) / 1000,
    category: getFeatureCategory(f.feature),
  }))

  const shapData24h = shap24h.slice(0, 12).map((f) => ({
    name: formatFeatureName(f.feature),
    rawName: f.feature,
    value: Math.round(f.mean_abs_shap * 1000) / 1000,
    category: getFeatureCategory(f.feature),
  }))

  return (
    <div className="glass-panel rounded-2xl p-5 sm:p-6">
      {/* Header with Model Architecture Telemetry */}
      <div className="flex flex-col gap-4 border-b border-border/50 pb-5 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 text-primary border border-primary/20">
            <BrainCircuit className="h-5 w-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-base font-bold text-foreground font-heading">
                Model Explainability & Benchmark Suite
              </h3>
              <span className="rounded-full bg-emerald-500/10 px-2.5 py-0.5 text-[11px] font-semibold text-emerald-400 border border-emerald-500/20">
                Production Ready
              </span>
            </div>
            <p className="text-xs text-muted-foreground mt-0.5">
              Strategy: <strong className="text-foreground capitalize">{data.model_architecture.strategy.replace(/_/g, " ")}</strong> · {data.model_architecture.features_used} Engineered Features · Hopsworks Feature Store
            </p>
          </div>
        </div>

        {/* Quick Architecture KPI Pills */}
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <div className="flex items-center gap-1.5 rounded-lg border border-border/60 bg-secondary/50 px-3 py-1.5">
            <Cpu className="h-3.5 w-3.5 text-primary" />
            <span className="text-muted-foreground">Horizons:</span>
            <span className="font-bold text-foreground font-mono">1h, 6h, 12h, 24h, 48h, 72h</span>
          </div>
          {report.split && (
            <div className="flex items-center gap-1.5 rounded-lg border border-border/60 bg-secondary/50 px-3 py-1.5">
              <Layers className="h-3.5 w-3.5 text-primary" />
              <span className="text-muted-foreground">Samples:</span>
              <span className="font-bold text-foreground font-mono">{(report.split.train + report.split.val + report.split.test).toLocaleString()}</span>
            </div>
          )}
        </div>
      </div>

      {/* Tabs Container */}
      <div className="mt-5">
        <Tabs defaultValue="metrics" className="w-full">
          <TabsList className="bg-secondary/70 p-1 rounded-xl border border-border/60">
            <TabsTrigger value="metrics" className="rounded-lg text-xs font-semibold data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              Evaluation Metrics
            </TabsTrigger>
            <TabsTrigger value="shap-1h" className="rounded-lg text-xs font-semibold data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              +1h SHAP Drivers
            </TabsTrigger>
            <TabsTrigger value="shap-24h" className="rounded-lg text-xs font-semibold data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              +24h SHAP Drivers
            </TabsTrigger>
            <TabsTrigger value="eda" className="rounded-lg text-xs font-semibold data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              Atmospheric Insights
            </TabsTrigger>
          </TabsList>

          {/* TAB 1: Evaluation Benchmark Table */}
          <TabsContent value="metrics" className="mt-4 space-y-4">
            <div className="overflow-x-auto rounded-xl border border-border/60 bg-secondary/20">
              <Table>
                <TableHeader className="bg-secondary/50">
                  <TableRow className="border-border/60">
                    <TableHead className="font-semibold text-foreground text-xs">Horizon</TableHead>
                    <TableHead className="font-semibold text-foreground text-xs">Winning Model</TableHead>
                    <TableHead className="text-right font-semibold text-foreground text-xs">RMSE (Error)</TableHead>
                    <TableHead className="text-right font-semibold text-foreground text-xs">MAE</TableHead>
                    <TableHead className="text-right font-semibold text-foreground text-xs">R² Goodness of Fit</TableHead>
                    <TableHead className="text-center font-semibold text-foreground text-xs">Reliability</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {Object.entries(horizons).map(([horizon, models]) => {
                    const best = report.best_model_per_horizon?.[horizon] || "XGBoost"
                    const bestMetrics = models[best]
                    if (!bestMetrics) return null
                    
                    const isSuperHighAccuracy = bestMetrics.r2 > 0.8
                    const isGoodAccuracy = bestMetrics.r2 > 0.4

                    return (
                      <TableRow key={horizon} className="border-border/40 hover:bg-secondary/40 transition-colors">
                        <TableCell className="font-bold text-foreground font-mono text-xs">
                          {horizon.toUpperCase()}
                        </TableCell>
                        <TableCell>
                          <span className="inline-flex items-center gap-1.5 rounded-md bg-primary/10 px-2 py-0.5 text-xs font-semibold text-primary border border-primary/20">
                            <BarChart3 className="h-3 w-3" />
                            {best}
                          </span>
                        </TableCell>
                        <TableCell className="text-right tabular-nums font-mono text-xs text-foreground font-medium">
                          {bestMetrics.rmse.toFixed(2)}
                        </TableCell>
                        <TableCell className="text-right tabular-nums font-mono text-xs text-foreground font-medium">
                          {bestMetrics.mae.toFixed(2)}
                        </TableCell>
                        <TableCell className="text-right tabular-nums">
                          <span
                            className={`inline-flex items-center rounded-md px-2 py-0.5 text-xs font-bold font-mono ${
                              isSuperHighAccuracy
                                ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
                                : isGoodAccuracy
                                ? "bg-amber-500/15 text-amber-400 border border-amber-500/30"
                                : "bg-rose-500/15 text-rose-400 border border-rose-500/30"
                            }`}
                          >
                            {bestMetrics.r2.toFixed(4)}
                          </span>
                        </TableCell>
                        <TableCell className="text-center">
                          <span className="text-[11px] text-muted-foreground font-medium">
                            {isSuperHighAccuracy ? "Exceptional" : isGoodAccuracy ? "High Confidence" : "Directional"}
                          </span>
                        </TableCell>
                      </TableRow>
                    )
                  })}
                </TableBody>
              </Table>
            </div>

            {report.split && (
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground px-1">
                <span>
                  Chronological Split — Train: <strong className="text-foreground">{report.split.train.toLocaleString()}</strong> (70%) · Val: <strong className="text-foreground">{report.split.val.toLocaleString()}</strong> (15%) · Test: <strong className="text-foreground">{report.split.test.toLocaleString()}</strong> (15%)
                </span>
                <span className="text-[11px] text-muted-foreground/70">
                  Zero data leakage across temporal sequence
                </span>
              </div>
            )}
          </TabsContent>

          {/* TAB 2: SHAP +1h */}
          <TabsContent value="shap-1h" className="mt-4 space-y-4">
            <div className="rounded-xl border border-primary/20 bg-primary/5 p-4 text-xs text-muted-foreground">
              <strong className="text-foreground">Short-Term (+1h) Feature Driving Dynamics:</strong> Near-term atmospheric prediction is governed by immediate air mass persistence (`us_aqi_lag_1h` and `pm2_5_rolling_24h`), capturing ground-level atmospheric inertia before synoptic weather patterns take over.
            </div>
            <ShapBarChart data={shapData1h} title="+1h Immediate Horizon" />
          </TabsContent>

          {/* TAB 3: SHAP +24h */}
          <TabsContent value="shap-24h" className="mt-4 space-y-4">
            <div className="rounded-xl border border-primary/20 bg-primary/5 p-4 text-xs text-muted-foreground">
              <strong className="text-foreground">Medium-Term (+24h) Feature Driving Dynamics:</strong> Over a full diurnal cycle, future numerical weather forecasts (`forecast_temp_humidity_24h`), cyclical temporal features (`month_sin`, `hour_sin`), and sustained multi-day PM2.5 trends surpass short-term lags.
            </div>
            <ShapBarChart data={shapData24h} title="+24h Diurnal Horizon" />
          </TabsContent>

          {/* TAB 4: EDA & Atmospheric Mechanics */}
          <TabsContent value="eda" className="mt-4">
            <div className="grid gap-4 md:grid-cols-2">
              <EdaInsightCard
                icon={<Sun className="h-5 w-5 text-amber-400" />}
                title="Diurnal Rush Hour Peaks"
                content="Karachi displays severe morning (07:00–10:00) and evening (18:00–22:00) spikes driven by vehicular commute emissions compounded by boundary layer subsidence trapping particulates near the surface."
              />
              <EdaInsightCard
                icon={<Wind className="h-5 w-5 text-sky-400" />}
                title="Arabian Sea Coastal Ventilation"
                content="Southwesterly sea breeze components (u/v vectors) provide natural ventilation flushing pollutants inland. When coastal winds stall, relative humidity rises and secondary aerosol nucleation accelerates."
              />
              <EdaInsightCard
                icon={<Flame className="h-5 w-5 text-rose-400" />}
                title="Winter Thermal Inversions"
                content="November through February exhibits elevated baseline AQI due to regional biomass burning and nocturnal radiation inversions that suppress vertical atmospheric dispersion."
              />
              <EdaInsightCard
                icon={<Sparkles className="h-5 w-5 text-purple-400" />}
                title="Multi-Model Architecture"
                content="XGBoost, Random Forest, and PyTorch LSTM models were benchmarked across 6 horizons. Gradient boosted trees deliver superior RMSE and lowest latency for real-time edge inference."
              />
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}

function ShapBarChart({ data, title }: { data: { name: string; rawName: string; value: number; category: string }[]; title: string }) {
  return (
    <div className="rounded-xl border border-border/60 bg-secondary/20 p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <span className="text-xs font-semibold text-foreground font-heading">
          Feature Importance (Mean Absolute SHAP Value |SHAP|) — {title}
        </span>
        <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
          <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-sky-400" /> Lag / History</span>
          <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-emerald-400" /> Weather Forecast</span>
          <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-purple-400" /> Cyclical Time</span>
        </div>
      </div>

      <div className="h-[380px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 5, right: 30, left: 140, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.25 0.018 255 / 0.4)" horizontal={false} />
            <XAxis type="number" tickLine={false} axisLine={false} stroke="oklch(0.65 0.02 255)" fontSize={10} />
            <YAxis
              type="category"
              dataKey="name"
              tickLine={false}
              axisLine={false}
              width={135}
              stroke="oklch(0.85 0.01 255)"
              fontSize={11}
            />
            <Tooltip 
              content={<CustomShapTooltip />} 
              cursor={{ fill: "rgba(255, 255, 255, 0.03)", radius: 4 }} 
            />
            <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={16}>
              {data.map((entry, index) => {
                const fill =
                  entry.category === "weather"
                    ? "#34D399"
                    : entry.category === "time"
                    ? "#C084FC"
                    : "#38BDF8"
                return <Cell key={`cell-${index}`} fill={fill} />
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function CustomShapTooltip({ active, payload }: any) {
  if (!active || !payload || !payload.length) return null
  const data = payload[0].payload

  return (
    <div className="glass-card rounded-xl border border-border/80 bg-popover/95 p-3 shadow-xl backdrop-blur-md text-xs">
      <p className="font-bold text-foreground">{data.name}</p>
      <p className="text-[10px] text-muted-foreground font-mono mt-0.5">`{data.rawName}`</p>
      <div className="mt-2 flex items-center justify-between gap-4 border-t border-border/40 pt-1.5">
        <span className="text-muted-foreground">Mean |SHAP| Impact:</span>
        <span className="font-bold font-mono text-primary">{data.value.toFixed(4)}</span>
      </div>
      <div className="mt-1 text-[10px] text-muted-foreground">
        Category: <strong className="text-foreground capitalize">{data.category}</strong>
      </div>
    </div>
  )
}

function EdaInsightCard({ icon, title, content }: { icon: React.ReactNode; title: string; content: string }) {
  return (
    <div className="glass-card rounded-xl p-4 flex flex-col justify-between">
      <div className="flex items-center gap-2.5 mb-2">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-secondary">
          {icon}
        </div>
        <h4 className="text-sm font-bold text-foreground font-heading">{title}</h4>
      </div>
      <p className="text-xs leading-relaxed text-muted-foreground/90">
        {content}
      </p>
    </div>
  )
}

function formatFeatureName(raw: string): string {
  return raw
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .replace("Us Aqi", "AQI")
    .replace("Pm2 5", "PM2.5")
    .replace("Pm10", "PM10")
    .replace("Lag 1h", "Lag 1h")
    .replace("Rolling 24h", "24h Avg")
}

function getFeatureCategory(raw: string): "lag" | "weather" | "time" {
  if (raw.includes("hour_") || raw.includes("month_") || raw.includes("day_") || raw.includes("sin") || raw.includes("cos")) {
    return "time"
  }
  if (raw.includes("temp") || raw.includes("humidity") || raw.includes("wind") || raw.includes("pressure") || raw.includes("precipitation")) {
    return "weather"
  }
  return "lag"
}
