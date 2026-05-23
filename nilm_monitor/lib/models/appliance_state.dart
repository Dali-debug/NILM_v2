class WindowApplianceInfo {
  final String state;
  final double powerW;
  final double confidence;

  const WindowApplianceInfo({
    required this.state,
    required this.powerW,
    required this.confidence,
  });

  factory WindowApplianceInfo.fromJson(Map<String, dynamic> json) =>
      WindowApplianceInfo(
        state: json['state'] as String? ?? 'OFF',
        powerW: (json['power_w'] as num? ?? 0).toDouble(),
        confidence: (json['confidence'] as num? ?? 0).toDouble(),
      );

  bool get isOn => state != 'OFF';
}

class WindowResult {
  final String status;
  final int samplesBuffered;
  final int windowSize;
  final int windowsCompleted;
  final int? windowIndex;
  final double? meanAggregate;
  final String? tStart;
  final String? tEnd;
  final Map<String, WindowApplianceInfo>? appliances;

  const WindowResult({
    required this.status,
    required this.samplesBuffered,
    required this.windowSize,
    required this.windowsCompleted,
    this.windowIndex,
    this.meanAggregate,
    this.tStart,
    this.tEnd,
    this.appliances,
  });

  bool get isReady => status == 'ready';
  bool get isCollecting => status == 'collecting';

  factory WindowResult.fromJson(Map<String, dynamic> json) {
    Map<String, WindowApplianceInfo>? apps;
    final appJson = json['appliances'] as Map<String, dynamic>?;
    if (appJson != null) {
      apps = {};
      appJson.forEach((key, value) {
        apps![key] = WindowApplianceInfo.fromJson(value as Map<String, dynamic>);
      });
    }
    return WindowResult(
      status: json['status'] as String? ?? 'collecting',
      samplesBuffered: json['samples_buffered'] as int? ?? 0,
      windowSize: json['window_size'] as int? ?? 7,
      windowsCompleted: json['windows_completed'] as int? ?? 0,
      windowIndex: json['window_index'] as int?,
      meanAggregate: (json['mean_aggregate'] as num?)?.toDouble(),
      tStart: json['t_start'] as String?,
      tEnd: json['t_end'] as String?,
      appliances: apps,
    );
  }
}
