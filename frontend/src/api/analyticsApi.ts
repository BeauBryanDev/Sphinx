// ============================================================
// Analytics API — endpoint wrappers for dashboard metrics
// ============================================================

import { request } from "@/services/http";
import type { AnalyticsMetric, TimeSeriesPoint } from "@/types";

export const fetchMetrics = (): Promise<AnalyticsMetric[]> => {
  return request<AnalyticsMetric[]>({
    method: "GET",
    url: "/v1/analytics/metrics",
  });
};

export const fetchActivitySeries = (
  range: "7d" | "30d" | "90d" = "30d",
): Promise<TimeSeriesPoint[]> => {
  return request<TimeSeriesPoint[]>({
    method: "GET",
    url: `/v1/analytics/activity?range=${range}`,
  });
};
