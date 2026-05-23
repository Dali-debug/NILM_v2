class HistoryEntry {
  final int? id;
  final int timestampMs;
  final double meanAggregate;
  final int windowIndex;
  final String tStart;
  final String tEnd;
  final String kettleState;
  final String microwaveState;
  final String fridgeState;
  final String tvState;
  final double kettlePower;
  final double microwavePower;
  final double fridgePower;
  final double tvPower;
  final double kettleConf;
  final double microwaveConf;
  final double fridgeConf;
  final double tvConf;

  const HistoryEntry({
    this.id,
    required this.timestampMs,
    required this.meanAggregate,
    required this.windowIndex,
    required this.tStart,
    required this.tEnd,
    required this.kettleState,
    required this.microwaveState,
    required this.fridgeState,
    required this.tvState,
    required this.kettlePower,
    required this.microwavePower,
    required this.fridgePower,
    required this.tvPower,
    required this.kettleConf,
    required this.microwaveConf,
    required this.fridgeConf,
    required this.tvConf,
  });

  DateTime get timestamp => DateTime.fromMillisecondsSinceEpoch(timestampMs);

  bool get kettleOn    => kettleState    != 'OFF';
  bool get microwaveOn => microwaveState != 'OFF';
  bool get fridgeOn    => fridgeState    != 'OFF';
  bool get tvOn        => tvState        != 'OFF';

  bool isOn(String appliance) {
    return switch (appliance) {
      'kettle'    => kettleOn,
      'microwave' => microwaveOn,
      'fridge'    => fridgeOn,
      'tv'        => tvOn,
      _           => false,
    };
  }

  Map<String, dynamic> toMap() => {
        'timestamp_ms':    timestampMs,
        'mean_aggregate':  meanAggregate,
        'window_index':    windowIndex,
        't_start':         tStart,
        't_end':           tEnd,
        'kettle_state':    kettleState,
        'microwave_state': microwaveState,
        'fridge_state':    fridgeState,
        'tv_state':        tvState,
        'kettle_power':    kettlePower,
        'microwave_power': microwavePower,
        'fridge_power':    fridgePower,
        'tv_power':        tvPower,
        'kettle_conf':     kettleConf,
        'microwave_conf':  microwaveConf,
        'fridge_conf':     fridgeConf,
        'tv_conf':         tvConf,
      };

  factory HistoryEntry.fromMap(Map<String, dynamic> m) => HistoryEntry(
        id:             m['id'] as int?,
        timestampMs:    m['timestamp_ms'] as int,
        meanAggregate:  (m['mean_aggregate'] as num).toDouble(),
        windowIndex:    m['window_index'] as int,
        tStart:         m['t_start'] as String,
        tEnd:           m['t_end'] as String,
        kettleState:    m['kettle_state'] as String,
        microwaveState: m['microwave_state'] as String,
        fridgeState:    m['fridge_state'] as String,
        tvState:        m['tv_state'] as String,
        kettlePower:    (m['kettle_power'] as num).toDouble(),
        microwavePower: (m['microwave_power'] as num).toDouble(),
        fridgePower:    (m['fridge_power'] as num).toDouble(),
        tvPower:        (m['tv_power'] as num).toDouble(),
        kettleConf:     (m['kettle_conf'] as num).toDouble(),
        microwaveConf:  (m['microwave_conf'] as num).toDouble(),
        fridgeConf:     (m['fridge_conf'] as num).toDouble(),
        tvConf:         (m['tv_conf'] as num).toDouble(),
      );
}

// Nominal power (W) per appliance for energy estimation
const Map<String, double> kNominalPower = {
  'kettle':    2000,
  'microwave': 1200,
  'fridge':    150,
  'tv':        100,
};

// Consistent colors per appliance across the whole app
const Map<String, int> kApplianceColorValues = {
  'kettle':    0xFF00E5A0,
  'microwave': 0xFFA855F7,
  'fridge':    0xFF60A5FA,
  'tv':        0xFFF59E0B,
};

const Map<String, String> kApplianceIcons = {
  'kettle':    '🫖',
  'microwave': '📟',
  'fridge':    '❄️',
  'tv':        '📺',
};

const Map<String, String> kApplianceNames = {
  'kettle':    'Kettle',
  'microwave': 'Microwave',
  'fridge':    'Fridge',
  'tv':        'TV',
};

const List<String> kAppliances = ['kettle', 'microwave', 'fridge', 'tv'];
