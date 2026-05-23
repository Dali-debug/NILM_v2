import 'package:flutter/material.dart';
import '../models/history_entry.dart';

const _bg      = Color(0xFF16162A);
const _border  = Color(0xFF2A2A4A);
const _textDim = Color(0xFF8888AA);

class UsageStatsCard extends StatelessWidget {
  final Map<String, double> runtimeSec;
  final Map<String, int>    activations;
  final Map<String, double> avgSessionSec;

  const UsageStatsCard({
    super.key,
    required this.runtimeSec,
    required this.activations,
    required this.avgSessionSec,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 14),
      decoration: BoxDecoration(
        color: _bg,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: _border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Usage Statistics',
              style: TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.w700,
                  fontSize: 14)),
          const SizedBox(height: 2),
          const Text('Runtime, activations & session duration',
              style: TextStyle(color: _textDim, fontSize: 11)),
          const SizedBox(height: 14),
          // Header row
          Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: Row(
              children: [
                const SizedBox(width: 80),
                _headerCell('Runtime'),
                _headerCell('Activations'),
                _headerCell('Avg session'),
              ],
            ),
          ),
          const Divider(color: Color(0xFF2A2A4A), height: 1),
          const SizedBox(height: 8),
          ...kAppliances.map((a) => _StatRow(
                appliance: a,
                runtimeSec: runtimeSec[a] ?? 0,
                activations: activations[a] ?? 0,
                avgSessionSec: avgSessionSec[a] ?? 0,
              )),
        ],
      ),
    );
  }

  Widget _headerCell(String text) => Expanded(
        child: Text(text,
            textAlign: TextAlign.center,
            style: const TextStyle(
                color: _textDim,
                fontSize: 9,
                fontWeight: FontWeight.w700,
                letterSpacing: 0.6)),
      );
}

class _StatRow extends StatelessWidget {
  final String appliance;
  final double runtimeSec;
  final int    activations;
  final double avgSessionSec;

  const _StatRow({
    required this.appliance,
    required this.runtimeSec,
    required this.activations,
    required this.avgSessionSec,
  });

  String _fmtDuration(double seconds) {
    if (seconds < 60)   return '${seconds.toInt()}s';
    if (seconds < 3600) return '${(seconds / 60).toStringAsFixed(1)}m';
    return '${(seconds / 3600).toStringAsFixed(1)}h';
  }

  @override
  Widget build(BuildContext context) {
    final color = Color(kApplianceColorValues[appliance]!);

    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        children: [
          SizedBox(
            width: 80,
            child: Row(
              children: [
                Text(kApplianceIcons[appliance]!,
                    style: const TextStyle(fontSize: 14)),
                const SizedBox(width: 6),
                Text(kApplianceNames[appliance]!,
                    style: const TextStyle(
                        color: Colors.white70,
                        fontSize: 11,
                        fontWeight: FontWeight.w600)),
              ],
            ),
          ),
          _valueCell(_fmtDuration(runtimeSec), color),
          _valueCell('$activations', color),
          _valueCell(
              activations > 0 ? _fmtDuration(avgSessionSec) : '—', color),
        ],
      ),
    );
  }

  Widget _valueCell(String text, Color color) => Expanded(
        child: Text(text,
            textAlign: TextAlign.center,
            style: TextStyle(
                color: color, fontSize: 12, fontWeight: FontWeight.w700)),
      );
}
