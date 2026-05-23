import 'package:flutter/material.dart';
import '../models/history_entry.dart';
import '../services/database_service.dart';
import '../widgets/power_line_chart.dart';
import '../widgets/appliance_timeline.dart';
import '../widgets/energy_bar_chart.dart';
import '../widgets/usage_stats_card.dart';

const _bgColor      = Color(0xFF0D0D1A);
const _surfaceColor = Color(0xFF16162A);
const _borderColor  = Color(0xFF2A2A4A);
const _accentColor  = Color(0xFF7C3AED);
const _accent2      = Color(0xFFA855F7);
const _onColor      = Color(0xFF00E5A0);
const _textColor    = Color(0xFFE2E2F0);
const _textDim      = Color(0xFF8888AA);

enum _Range { h1, h24, d7, d30 }

extension _RangeExt on _Range {
  String get label => switch (this) {
        _Range.h1  => '1H',
        _Range.h24 => '24H',
        _Range.d7  => '7D',
        _Range.d30 => '30D',
      };

  DateTime get from => switch (this) {
        _Range.h1  => DateTime.now().subtract(const Duration(hours: 1)),
        _Range.h24 => DateTime.now().subtract(const Duration(hours: 24)),
        _Range.d7  => DateTime.now().subtract(const Duration(days: 7)),
        _Range.d30 => DateTime.now().subtract(const Duration(days: 30)),
      };
}

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen>
    with SingleTickerProviderStateMixin {
  final _db = DatabaseService();

  _Range _range   = _Range.h24;
  bool   _loading = true;

  List<HistoryEntry>     _entries    = [];
  Map<String, double>    _energyKwh  = {};
  Map<String, double>    _runtime    = {};
  Map<String, int>       _acts       = {};
  Map<String, double>    _avgSession = {};

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() => _loading = true);
    final entries = await _db.getByTimeRange(_range.from);
    setState(() {
      _entries    = entries;
      _energyKwh  = _db.computeEnergyKwh(entries);
      _runtime    = _db.computeRuntimeSeconds(entries);
      _acts       = _db.computeActivations(entries);
      _avgSession = _db.computeAvgSession(entries);
      _loading    = false;
    });
  }

  Future<void> _clearHistory() async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        backgroundColor: _surfaceColor,
        title: const Text('Clear History',
            style: TextStyle(color: _textColor)),
        content: const Text(
            'This will delete all stored readings from the device.',
            style: TextStyle(color: _textDim)),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancel',
                  style: TextStyle(color: _textDim))),
          TextButton(
              onPressed: () => Navigator.pop(context, true),
              child: const Text('Clear',
                  style: TextStyle(color: Colors.redAccent))),
        ],
      ),
    );
    if (confirm == true) {
      await _db.clearAll();
      _load();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _bgColor,
      appBar: AppBar(
        backgroundColor: _surfaceColor,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new,
              color: _textColor, size: 18),
          onPressed: () => Navigator.pop(context),
        ),
        title: const Text('History & Charts',
            style: TextStyle(
                color: _textColor,
                fontWeight: FontWeight.w700,
                fontSize: 18)),
        actions: [
          IconButton(
            icon: const Icon(Icons.delete_outline,
                color: _textDim, size: 20),
            tooltip: 'Clear history',
            onPressed: _clearHistory,
          ),
        ],
      ),
      body: Column(
        children: [
          _buildRangeSelector(),
          Expanded(
            child: _loading
                ? const Center(
                    child: CircularProgressIndicator(color: _accent2))
                : _entries.isEmpty
                    ? _buildEmptyState()
                    : _buildCharts(),
          ),
        ],
      ),
    );
  }

  // ── Range selector ─────────────────────────────────────────────────
  Widget _buildRangeSelector() {
    return Container(
      color: _surfaceColor,
      padding: const EdgeInsets.fromLTRB(16, 10, 16, 12),
      child: Row(
        children: _Range.values.map((r) {
          final active = r == _range;
          return Expanded(
            child: GestureDetector(
              onTap: () {
                setState(() => _range = r);
                _load();
              },
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 200),
                margin: const EdgeInsets.symmetric(horizontal: 3),
                padding: const EdgeInsets.symmetric(vertical: 8),
                decoration: BoxDecoration(
                  color: active ? _accentColor : Colors.transparent,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(
                      color: active ? _accent2 : _borderColor),
                ),
                child: Center(
                  child: Text(r.label,
                      style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w700,
                          color: active ? Colors.white : _textDim)),
                ),
              ),
            ),
          );
        }).toList(),
      ),
    );
  }

  // ── Charts ──────────────────────────────────────────────────────────
  Widget _buildCharts() {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        PowerLineChart(entries: _entries),
        ApplianceTimeline(entries: _entries),
        EnergyBarChart(energyKwh: _energyKwh),
        UsageStatsCard(
          runtimeSec:    _runtime,
          activations:   _acts,
          avgSessionSec: _avgSession,
        ),
        // Summary chip
        _buildSummaryChip(),
        const SizedBox(height: 8),
      ],
    );
  }

  Widget _buildSummaryChip() {
    final totalKwh = _energyKwh.values.fold(0.0, (a, b) => a + b);
    final totalActs = _acts.values.fold(0, (a, b) => a + b);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: _onColor.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: _onColor.withValues(alpha: 0.3)),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _summaryItem('Readings', '${_entries.length}', _onColor),
          _summaryItem('Total kWh',
              totalKwh.toStringAsFixed(3), _onColor),
          _summaryItem('Activations', '$totalActs', _onColor),
        ],
      ),
    );
  }

  Widget _summaryItem(String label, String value, Color color) => Column(
        children: [
          Text(value,
              style: TextStyle(
                  color: color,
                  fontSize: 16,
                  fontWeight: FontWeight.w800)),
          const SizedBox(height: 2),
          Text(label,
              style: const TextStyle(color: _textDim, fontSize: 10)),
        ],
      );

  // ── Empty state ─────────────────────────────────────────────────────
  Widget _buildEmptyState() {
    return const Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text('📊', style: TextStyle(fontSize: 48)),
          SizedBox(height: 16),
          Text('No data yet',
              style: TextStyle(
                  color: _textColor,
                  fontSize: 18,
                  fontWeight: FontWeight.w700)),
          SizedBox(height: 8),
          Text(
              'Start monitoring to see your history\nfor the selected time range',
              textAlign: TextAlign.center,
              style: TextStyle(color: _textDim, fontSize: 13)),
        ],
      ),
    );
  }
}
