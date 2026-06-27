// ============================================================
// <ActivityChart /> — Recharts-powered activity line chart
// ============================================================

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card } from "@/components/common";
import { useAnalytics } from "@/hooks/useAnalytics";

export const ActivityChart = () => {
  const { series, isLoading } = useAnalytics("7d");

  return (
    <Card title="Weekly Activity" subtitle="Queries vs. decodings" icon="📊">
      {isLoading ? (
        <div className="flex h-64 items-center justify-center text-amber-400/70">
          Loading chart data…
        </div>
      ) : (
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={series}
              margin={{ top: 10, right: 20, left: 0, bottom: 0 }}
            >
              <CartesianGrid stroke="#b45309" strokeOpacity={0.25} />
              <XAxis
                dataKey="name"
                stroke="#f59e0b"
                fontSize={12}
                tickLine={false}
              />
              <YAxis
                stroke="#f59e0b"
                fontSize={12}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip
                contentStyle={{
                  background: "rgba(28, 20, 10, 0.95)",
                  border: "1px solid rgba(180, 120, 30, 0.5)",
                  borderRadius: 6,
                  color: "#fde68a",
                  fontSize: 12,
                }}
              />
              <Legend wrapperStyle={{ color: "#fcd34d", fontSize: 12 }} />
              <Line
                type="monotone"
                dataKey="queries"
                stroke="#fbbf24"
                strokeWidth={2.5}
                dot={{ r: 3, fill: "#fbbf24" }}
                activeDot={{ r: 5 }}
              />
              <Line
                type="monotone"
                dataKey="decodings"
                stroke="#f97316"
                strokeWidth={2.5}
                dot={{ r: 3, fill: "#f97316" }}
                activeDot={{ r: 5 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </Card>
  );
};

export default ActivityChart;
