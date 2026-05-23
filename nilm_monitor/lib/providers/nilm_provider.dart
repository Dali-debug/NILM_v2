import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/appliance_state.dart';
import '../models/history_entry.dart';
import '../services/nilm_api.dart';
import '../services/database_service.dart';

class NilmProvider extends ChangeNotifier {
  final _db = DatabaseService();

  String _ip    = '192.168.1.42';
  int    _port  = 8080;
  int    _house = 3;

  bool          _streaming        = false;
  String?       _error;
  WindowResult? _lastResult;
  int           _samplesBuffered  = 0;
  int           _windowSize       = 7;
  int           _windowsCompleted = 0;
  final List<WindowResult> _history = [];

  StreamSubscription<SseEvent>? _sseSub;

  String        get ip               => _ip;
  int           get port             => _port;
  int           get house            => _house;
  bool          get isStreaming      => _streaming;
  String?       get error            => _error;
  WindowResult? get lastResult       => _lastResult;
  int           get samplesBuffered  => _samplesBuffered;
  int           get windowSize       => _windowSize;
  int           get windowsCompleted => _windowsCompleted;
  bool          get isReady          => _lastResult?.isReady ?? false;
  double?       get meanAggregate    => _lastResult?.meanAggregate;
  List<WindowResult> get history     => List.unmodifiable(_history);

  NilmApi get _api => NilmApi(ip: _ip, port: _port);

  NilmProvider() {
    _loadPrefs();
  }

  Future<void> _loadPrefs() async {
    final prefs = await SharedPreferences.getInstance();
    _ip    = prefs.getString('pi_ip')  ?? '192.168.1.42';
    _port  = prefs.getInt('pi_port')   ?? 8080;
    _house = prefs.getInt('pi_house')  ?? 3;
    notifyListeners();
  }

  Future<void> saveSettings(String ip, int port, int house) async {
    _ip    = ip;
    _port  = port;
    _house = house;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('pi_ip',  ip);
    await prefs.setInt('pi_port',   port);
    await prefs.setInt('pi_house',  house);
    notifyListeners();
  }

  Future<void> setHouse(int house) async {
    if (_streaming) await stopStream();
    _house = house;
    notifyListeners();
  }

  // ── Streaming ────────────────────────────────────────────────────────

  Future<void> startStream() async {
    if (_streaming) return;
    _error    = null;
    _streaming = true;
    notifyListeners();

    try {
      await _api.startStream(_house);
    } catch (e) {
      _error     = e.toString().replaceFirst('Exception: ', '');
      _streaming = false;
      notifyListeners();
      return;
    }

    _sseSub = _api.connectSSE().listen(
      _onSseEvent,
      onDone: () {
        _streaming = false;
        notifyListeners();
      },
      onError: (e) {
        _error     = 'SSE: ${e.toString()}';
        _streaming = false;
        notifyListeners();
      },
    );
  }

  Future<void> stopStream() async {
    await _sseSub?.cancel();
    _sseSub    = null;
    _streaming = false;
    try {
      await _api.stopStream(_house);
    } catch (_) {}
    notifyListeners();
  }

  void _onSseEvent(SseEvent event) {
    final d = event.data;
    if ((d['house'] as int?) != _house) return;

    if (event.type == 'status') {
      _samplesBuffered  = d['samples_buffered']  as int? ?? _samplesBuffered;
      _windowSize       = d['window_size']        as int? ?? _windowSize;
      _windowsCompleted = d['windows_completed']  as int? ?? _windowsCompleted;
      notifyListeners();

    } else if (event.type == 'result') {
      final res         = WindowResult.fromJson(d);
      _lastResult       = res;
      _samplesBuffered  = res.samplesBuffered;
      _windowSize       = res.windowSize;
      _windowsCompleted = res.windowsCompleted;

      _history.insert(0, res);
      if (_history.length > 10) _history.removeLast();

      final apps = res.appliances ?? {};
      _db.insert(HistoryEntry(
        timestampMs:    DateTime.now().millisecondsSinceEpoch,
        meanAggregate:  res.meanAggregate ?? 0,
        windowIndex:    res.windowIndex   ?? 0,
        tStart:         res.tStart        ?? '',
        tEnd:           res.tEnd          ?? '',
        kettleState:    apps['kettle']?.state    ?? 'OFF',
        microwaveState: apps['microwave']?.state ?? 'OFF',
        fridgeState:    apps['fridge']?.state    ?? 'OFF',
        tvState:        apps['tv']?.state        ?? 'OFF',
        kettlePower:    apps['kettle']?.powerW    ?? 0,
        microwavePower: apps['microwave']?.powerW ?? 0,
        fridgePower:    apps['fridge']?.powerW    ?? 0,
        tvPower:        apps['tv']?.powerW        ?? 0,
        kettleConf:     apps['kettle']?.confidence    ?? 0,
        microwaveConf:  apps['microwave']?.confidence ?? 0,
        fridgeConf:     apps['fridge']?.confidence    ?? 0,
        tvConf:         apps['tv']?.confidence        ?? 0,
      ));

      notifyListeners();
    }
  }

  // ── Other ────────────────────────────────────────────────────────────

  Future<void> reset() async {
    await stopStream();
    try {
      await _api.reset(_house);
    } catch (_) {}
    _lastResult       = null;
    _samplesBuffered  = 0;
    _windowsCompleted = 0;
    _history.clear();
    _error = null;
    notifyListeners();
  }

  Future<String?> testConnection() async {
    try {
      final houses = await _api.getHouses();
      return 'Connecté — maisons disponibles: $houses';
    } catch (e) {
      return 'Échec: ${e.toString().replaceFirst('Exception: ', '')}';
    }
  }

  @override
  void dispose() {
    _sseSub?.cancel();
    super.dispose();
  }
}
