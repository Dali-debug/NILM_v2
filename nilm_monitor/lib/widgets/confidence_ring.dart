import 'dart:math';
import 'package:flutter/material.dart';

class ConfidenceRing extends StatelessWidget {
  final double value; // 0.0 → 1.0
  final Color color;
  final double size;

  const ConfidenceRing({
    super.key,
    required this.value,
    required this.color,
    this.size = 36,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: size,
      height: size,
      child: CustomPaint(
        painter: _RingPainter(value: value, color: color),
        child: Center(
          child: Text(
            '${(value * 100).round()}%',
            style: TextStyle(
              fontSize: size * 0.22,
              fontWeight: FontWeight.w700,
              color: color,
            ),
          ),
        ),
      ),
    );
  }
}

class _RingPainter extends CustomPainter {
  final double value;
  final Color color;

  _RingPainter({required this.value, required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final cx = size.width / 2;
    final cy = size.height / 2;
    final radius = (size.width / 2) - 3;
    const strokeW = 3.0;

    // Track
    canvas.drawArc(
      Rect.fromCircle(center: Offset(cx, cy), radius: radius),
      0, 2 * pi, false,
      Paint()
        ..color = color.withOpacity(0.15)
        ..style = PaintingStyle.stroke
        ..strokeWidth = strokeW,
    );

    // Progress
    canvas.drawArc(
      Rect.fromCircle(center: Offset(cx, cy), radius: radius),
      -pi / 2,
      2 * pi * value,
      false,
      Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = strokeW
        ..strokeCap = StrokeCap.round,
    );
  }

  @override
  bool shouldRepaint(_RingPainter old) => old.value != value || old.color != color;
}
