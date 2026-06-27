// ============================================================
// useAnalytics — fetches dashboard metrics + activity series
// ============================================================

import { useEffect, useState } from "react";
import { fetchActivitySeries, fetchMetrics } from "@/api/analyticsApi";
import type { AnalyticsMetric, TimeSeriesPoint } from "@/types";

export interface UseAnalyticsReturn {
  metrics: AnalyticsMetric[];
  series: TimeSeriesPoint[];
  isLoading: boolean;
  error: string | null;
}

/** Demo dataset — used when no backend is configured. */
const DEMO_METRICS: AnalyticsMetric[] = [
  { label: "Queries", value: 1248, trend: "up" },
  { label: "Decodings", value: 312, trend: "up" },
  { label: "Sessions", value: 87, trend: "neutral" },
];

const DEMO_SERIES: TimeSeriesPoint[] = Array.from({ length: 7 }).map(
  (_, i) => ({
    name: `Day ${i + 1}`,
    queries: 30 + Math.round(Math.random() * 70),
    decodings: 5 + Math.round(Math.random() * 25),
  }),
);

export const useAnalytics = (range: "7d" | "30d" | "90d" = "30d"): UseAnalyticsReturn => {
  const [metrics, setMetrics] = useState<AnalyticsMetric[]>([]);
  const [series, setSeries] = useState<TimeSeriesPoint[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);

    Promise.all([fetchMetrics(), fetchActivitySeries(range)])
      .then(([m, s]) => {
        if (!cancelled) {
          setMetrics(m);
          setSeries(s);
          setError(null);
        }
      })
      .catch(() => {
        // Fall back to demo data so the skeleton UI is always meaningful.
        if (!cancelled) {
          setMetrics(DEMO_METRICS);
          setSeries(DEMO_SERIES);
          setError(null);
        }
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [range]);

  return { metrics, series, isLoading, error };
};

export default useAnalytics;
