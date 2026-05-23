import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import '../models/history_entry.dart';

const _bg         = Color(0xFF16162A);
const _border     = Color(0xFF2A2A4A);
const _textDim    = Color(0xFF8888AA);
const _lineColor  = Color(0xFF00E5A0);

class PowerLineChart extends StatelessWidget {
  final List<HistoryEntry> entries;

  const PowerLineChart({super.key, required this.entries});

  @override
  Widget build(BuildContext context) {
    if (entries.isEmpty) return const SizedBox.shrink();

    final minTs = entries.first.timestampMs.toDouble();
    final spots = <FlSpot>[];

    for (final e in entries) {
      final x = (e.timestampMs - minTs) / 60000; // minutes
      spots.add(FlSpot(x, e.meanAggregate));
    }

    final maxY = entries.map((e) => e.meanAggregate).reduce((a, b) => a > b ? a : b);

    return _ChartCard(
      title: 'Total Power',
      subtitle: 'Mean aggregate per window (W)',
      child: LineChart(
        LineChartData(
          backgroundColor: _bg,
          minY: 0,
          maxY: maxY * 1.15,
          clipData: const FlClipData.all(),
          gridData: FlGridData(
            show: true,
            drawVerticalLine: false,
            getDrawingHorizontalLine: (_) => const FlLine(
              color: _border,
              strokeWidth: 1,
            ),
          ),
          borderData: FlBorderData(show: false),
          titlesData: FlTitlesData(
            leftTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                reservedSize: 44,
                getTitlesWidget: (v, _) => Text(
                  '${v.toInt()}W',
                  style: const TextStyle(color: _textDim, fontSize: 9),
                ),
              ),
            ),
            bottomTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                reservedSize: 22,
                interval: _xInterval(spots),
                getTitlesWidget: (v, _) => Text(
                  '${v.toInt()}m',
                  style: const TextStyle(color: _textDim, fontSize: 9),
                ),
              ),
            ),
            topTitles:   const AxisTitles(sideTitles: SideTitles(showTitles: false)),
            rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          ),
          lineBarsData: [
            LineChartBarData(
              spots: spots,
              isCurved: true,
              color: _lineColor,
              barWidth: 2.5,
              dotData: const FlDotData(show: false),
              belowBarData: BarAreaData(
                show: true,
                color: _lineColor.withValues(alpha: 0.08),
              ),
            ),
          ],
          lineTouchData: LineTouchData(
            touchTooltipData: LineTouchTooltipData(
              getTooltipColor: (_) => const Color(0xFF1F1F3A),
              getTooltipItems: (spots) => spots.map((s) => LineTooltipItem(
                '${s.y.toStringAsFixed(0)} W',
                const TextStyle(
                  color: _lineColor,
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                ),
              )).toList(),
            ),
          ),
        ),
        duration: const Duration(milliseconds: 400),
      ),
    );
  }

  double _xInterval(List<FlSpot> spots) {
    if (spots.isEmpty) return 1;
    final range = spots.last.x - spots.first.x;
    if (range <= 0) return 1;
    return (range / 5).ceilToDouble().clamp(1, 9999);
  }
}

// ── Shared card wrapper ────────────────────────────────────────────────────
class _ChartCard extends StatelessWidget {
  final String title;
  final String subtitle;
  final Widget child;

  const _ChartCard({
    required this.title,
    required this.subtitle,
    required this.child,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.fromLTRB(16, 14, 12, 14),
      decoration: BoxDecoration(
        color: _bg,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: _border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title,
              style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.w700,
                  fontSize: 14)),
          const SizedBox(height: 2),
          Text(subtitle,
              style: const TextStyle(color: _textDim, fontSize: 11)),
          const SizedBox(height: 14),
          SizedBox(height: 220, child: child),
        ],
      ),
    );
  }
}
