import 'package:flutter/material.dart';
import '../models/history_entry.dart';

const _bg      = Color(0xFF16162A);
const _border  = Color(0xFF2A2A4A);
const _textDim = Color(0xFF8888AA);
const _offColor = Color(0xFF2A2A4A);

class ApplianceTimeline extends StatelessWidget {
  final List<HistoryEntry> entries;

  const ApplianceTimeline({super.key, required this.entries});

  @override
  Widget build(BuildContext context) {
    if (entries.isEmpty) return const SizedBox.shrink();

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
          const Text('Appliance Timeline',
              style: TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.w700,
                  fontSize: 14)),
          const SizedBox(height: 2),
          const Text('ON/OFF activity over selected period',
              style: TextStyle(color: _textDim, fontSize: 11)),
          const SizedBox(height: 14),
          ...kAppliances.map((a) => _ApplianceRow(
                appliance: a,
                entries: entries,
              )),
        ],
      ),
    );
  }
}

class _ApplianceRow extends StatelessWidget {
  final String appliance;
  final List<HistoryEntry> entries;

  const _ApplianceRow({required this.appliance, required this.entries});

  @override
  Widget build(BuildContext context) {
    final color = Color(kApplianceColorValues[appliance]!);

    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        children: [
          SizedBox(
            width: 72,
            child: Row(
              children: [
                Text(kApplianceIcons[appliance]!, style: const TextStyle(fontSize: 14)),
                const SizedBox(width: 5),
                Text(
                  kApplianceNames[appliance]!,
                  style: const TextStyle(
                      color: _textDim, fontSize: 10, fontWeight: FontWeight.w600),
                  overflow: TextOverflow.ellipsis,
                ),
              ],
            ),
          ),
          Expanded(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: SizedBox(
                height: 18,
                child: CustomPaint(
                  painter: _TimelinePainter(
                    entries: entries,
                    appliance: appliance,
                    onColor: color,
                    offColor: _offColor,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _TimelinePainter extends CustomPainter {
  final List<HistoryEntry> entries;
  final String appliance;
  final Color onColor;
  final Color offColor;

  _TimelinePainter({
    required this.entries,
    required this.appliance,
    required this.onColor,
    required this.offColor,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (entries.length < 2) return;

    final minTs = entries.first.timestampMs.toDouble();
    final maxTs = entries.last.timestampMs.toDouble();
    final range = maxTs - minTs;
    if (range <= 0) return;

    // Background
    canvas.drawRRect(
      RRect.fromRectAndRadius(
          Offset.zero & size, const Radius.circular(4)),
      Paint()..color = offColor,
    );

    // ON segments
    final paint = Paint()..color = onColor.withValues(alpha: 0.85);

    for (int i = 0; i < entries.length - 1; i++) {
      if (!entries[i].isOn(appliance)) continue;

      final x1 = ((entries[i].timestampMs - minTs) / range) * size.width;
      final x2 = ((entries[i + 1].timestampMs - minTs) / range) * size.width;
      canvas.drawRect(Rect.fromLTWH(x1, 0, (x2 - x1).clamp(1.0, size.width), size.height), paint);
    }
  }

  @override
  bool shouldRepaint(_TimelinePainter old) =>
      old.entries != entries || old.appliance != appliance;
}
