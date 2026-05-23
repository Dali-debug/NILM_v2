import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import '../models/appliance_state.dart';

class SseEvent {
  final String type;
  final Map<String, dynamic> data;
  const SseEvent({required this.type, required this.data});
}

class NilmApi {
  final String ip;
  final int port;

  NilmApi({required this.ip, required this.port});

  String get _base => 'http://$ip:$port';

  // ── Auto-stream ──────────────────────────────────────────────────────

  Future<void> startStream(int house) async {
    await http
        .post(
          Uri.parse('$_base/start_stream'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({'house': house}),
        )
        .timeout(const Duration(seconds: 10));
  }

  Future<void> stopStream(int house) async {
    await http
        .post(
          Uri.parse('$_base/stop_stream'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({'house': house}),
        )
        .timeout(const Duration(seconds: 5));
  }

  /// Connects to GET /events and yields SSE events as they arrive.
  /// Each event has a [type] ('status' or 'result') and a [data] map.
  Stream<SseEvent> connectSSE() async* {
    final client = HttpClient();
    try {
      final request = await client.getUrl(Uri.parse('$_base/events'));
      request.headers.set('Accept', 'text/event-stream');
      request.headers.set('Cache-Control', 'no-cache');
      final response = await request.close();

      String buffer    = '';
      String eventType = 'message';

      await for (final chunk in response.transform(utf8.decoder)) {
        buffer += chunk;
        final lines = buffer.split('\n');
        buffer = lines.last; // keep incomplete trailing line

        for (final line in lines.sublist(0, lines.length - 1)) {
          if (line.startsWith('event: ')) {
            eventType = line.substring(7).trim();
          } else if (line.startsWith('data: ')) {
            final raw = line.substring(6).trim();
            if (raw.isNotEmpty && raw != 'connected') {
              try {
                final json = jsonDecode(raw) as Map<String, dynamic>;
                yield SseEvent(type: eventType, data: json);
              } catch (_) {}
            }
            eventType = 'message';
          }
        }
      }
    } finally {
      client.close();
    }
  }

  // ── Existing endpoints ───────────────────────────────────────────────

  Future<void> reset(int house) async {
    await http
        .post(
          Uri.parse('$_base/reset'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({'house': house}),
        )
        .timeout(const Duration(seconds: 5));
  }

  Future<List<int>> getHouses() async {
    final response = await http
        .get(Uri.parse('$_base/houses'))
        .timeout(const Duration(seconds: 5));

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body) as Map<String, dynamic>;
      return List<int>.from(data['houses'] as List);
    }
    throw Exception('Connexion échouée (${response.statusCode})');
  }

  // kept for backward compat (settings test)
  Future<WindowResult> pushSample(double power, int house) async {
    final response = await http
        .post(
          Uri.parse('$_base/push_sample'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({'power': power, 'house': house}),
        )
        .timeout(const Duration(seconds: 10));

    if (response.statusCode == 200) {
      return WindowResult.fromJson(
          jsonDecode(response.body) as Map<String, dynamic>);
    }
    final err = jsonDecode(response.body);
    throw Exception(err['error'] ?? 'Erreur ${response.statusCode}');
  }
}
