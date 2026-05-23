import 'package:flutter/material.dart';
import '../models/appliance_state.dart';
import 'confidence_ring.dart';

const _bgColor      = Color(0xFF0D0D1A);
const _surfaceColor = Color(0xFF16162A);
const _borderColor  = Color(0xFF2A2A4A);
const _onColor      = Color(0xFF00E5A0);
const _lowColor     = Color(0xFFF59E0B);
const _offColor     = Color(0xFF3A3A5C);
const _textDim      = Color(0xFF8888AA);

const _icons = {
  'kettle':    '🫖',
  'microwave': '📟',
  'fridge':    '❄️',
  'tv':        '📺',
};

const _names = {
  'kettle':    'Kettle',
  'microwave': 'Microwave',
  'fridge':    'Fridge',
  'tv':        'TV',
};

class ApplianceCard extends StatelessWidget {
  final String name;
  final WindowApplianceInfo? info;

  const ApplianceCard({super.key, required this.name, this.info});

  Color get _stateColor {
    if (info == null) return _offColor;
    return switch (info!.state) {
      'ON'   => _onColor,
      'HIGH' => _onColor,
      'LOW'  => _lowColor,
      _      => _offColor,
    };
  }

  bool get _isActive => info != null && info!.isOn;

  @override
  Widget build(BuildContext context) {
    final color = _stateColor;

    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      decoration: BoxDecoration(
        color: _isActive ? color.withOpacity(0.07) : _surfaceColor,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: _isActive ? color.withOpacity(0.6) : _borderColor,
          width: 1,
        ),
        boxShadow: _isActive
            ? [BoxShadow(color: color.withOpacity(0.2), blurRadius: 16, spreadRadius: 1)]
            : [],
      ),
      padding: const EdgeInsets.all(14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                _icons[name] ?? '⚡',
                style: const TextStyle(fontSize: 22),
              ),
              Row(
                children: [
                  AnimatedContainer(
                    duration: const Duration(milliseconds: 300),
                    width: 7,
                    height: 7,
                    decoration: BoxDecoration(
                      color: color,
                      shape: BoxShape.circle,
                    ),
                  ),
                  const SizedBox(width: 4),
                  Text(
                    info?.state ?? '—',
                    style: TextStyle(
                      fontSize: 10,
                      fontWeight: FontWeight.w700,
                      color: color,
                      letterSpacing: 0.8,
                    ),
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 8),

          Text(
            _names[name] ?? name,
            style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w600,
              color: _isActive ? Colors.white70 : _textDim,
              letterSpacing: 0.4,
            ),
          ),
          const SizedBox(height: 4),

          Text(
            info != null ? '${info!.powerW.round()} W' : '— W',
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.w800,
              color: _isActive ? color : _textDim,
            ),
          ),

          const SizedBox(height: 10),

          Center(
            child: ConfidenceRing(
              value: info?.confidence ?? 0,
              color: color,
              size: 44,
            ),
          ),
        ],
      ),
    );
  }
}
