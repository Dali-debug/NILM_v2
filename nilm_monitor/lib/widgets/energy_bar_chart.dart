import 'package:flutter/material.dart';
import '../models/history_entry.dart';

const _bg      = Color(0xFF16162A);
const _border  = Color(0xFF2A2A4A);
const _textDim = Color(0xFF8888AA);

class EnergyBarChart extends StatelessWidget {
  final Map<String, double> energyKwh;

  const EnergyBarChart({super.key, required this.energyKwh});

  @override
  Widget build(BuildContext context) {
    final maxVal = energyKwh.values.fold(0.0, (a, b) => a > b ? a : b);

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
          const Text('Energy Consumption',
              style: TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.w700,
                  fontSize: 14)),
          const SizedBox(height: 2),
          const Text('Estimated kWh per appliance',
              style: TextStyle(color: _textDim, fontSize: 11)),
          const SizedBox(height: 16),
          ...kAppliances.map((a) => _EnergyRow(
                appliance: a,
                kwh: energyKwh[a] ?? 0,
                maxKwh: maxVal,
              )),
        ],
      ),
    );
  }
}

class _EnergyRow extends StatelessWidget {
  final String appliance;
  final double kwh;
  final double maxKwh;

  const _EnergyRow({
    required this.appliance,
    required this.kwh,
    required this.maxKwh,
  });

  @override
  Widget build(BuildContext context) {
    final color  = Color(kApplianceColorValues[appliance]!);
    final ratio  = maxKwh > 0 ? kwh / maxKwh : 0.0;

    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(kApplianceIcons[appliance]!, style: const TextStyle(fontSize: 14)),
              const SizedBox(width: 6),
              Text(kApplianceNames[appliance]!,
                  style: const TextStyle(
                      color: Colors.white70, fontSize: 12, fontWeight: FontWeight.w600)),
              const Spacer(),
              Text(
                kwh >= 0.001
                    ? '${kwh.toStringAsFixed(3)} kWh'
                    : '< 0.001 kWh',
                style: TextStyle(
                    color: color, fontSize: 12, fontWeight: FontWeight.w700),
              ),
            ],
          ),
          const SizedBox(height: 6),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: ratio.clamp(0.0, 1.0),
              minHeight: 8,
              backgroundColor: const Color(0xFF2A2A4A),
              valueColor: AlwaysStoppedAnimation<Color>(color),
            ),
          ),
        ],
      ),
    );
  }
}
