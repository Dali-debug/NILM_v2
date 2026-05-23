import 'package:path/path.dart';
import 'package:sqflite/sqflite.dart';
import '../models/history_entry.dart';

class DatabaseService {
  static final DatabaseService _instance = DatabaseService._();
  factory DatabaseService() => _instance;
  DatabaseService._();

  Database? _db;

  Future<Database> get db async {
    _db ??= await _init();
    return _db!;
  }

  Future<Database> _init() async {
    final dbPath = await getDatabasesPath();
    final path = join(dbPath, 'nilm_history.db');
    return openDatabase(
      path,
      version: 2,
      onCreate: _createV2,
      onUpgrade: (db, oldVersion, newVersion) async {
        if (oldVersion < 2) {
          await db.execute('DROP TABLE IF EXISTS readings');
          await _createV2(db, newVersion);
        }
      },
    );
  }

  Future<void> _createV2(Database db, int version) => db.execute('''
    CREATE TABLE readings (
      id              INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp_ms    INTEGER NOT NULL,
      mean_aggregate  REAL,
      window_index    INTEGER,
      t_start         TEXT,
      t_end           TEXT,
      kettle_state    TEXT,
      microwave_state TEXT,
      fridge_state    TEXT,
      tv_state        TEXT,
      kettle_power    REAL,
      microwave_power REAL,
      fridge_power    REAL,
      tv_power        REAL,
      kettle_conf     REAL,
      microwave_conf  REAL,
      fridge_conf     REAL,
      tv_conf         REAL
    )
  ''');

  Future<void> insert(HistoryEntry entry) async {
    final database = await db;
    await database.insert('readings', entry.toMap(),
        conflictAlgorithm: ConflictAlgorithm.replace);
  }

  Future<List<HistoryEntry>> getByTimeRange(DateTime from) async {
    final database = await db;
    final rows = await database.query(
      'readings',
      where: 'timestamp_ms >= ?',
      whereArgs: [from.millisecondsSinceEpoch],
      orderBy: 'timestamp_ms ASC',
    );
    return rows.map(HistoryEntry.fromMap).toList();
  }

  Future<void> clearAll() async {
    final database = await db;
    await database.delete('readings');
  }

  // ── Analytics helpers ─────────────────────────────────────────────

  /// Returns total ON duration in seconds per appliance.
  Map<String, double> computeRuntimeSeconds(List<HistoryEntry> entries) {
    final totals = {for (final a in kAppliances) a: 0.0};
    if (entries.length < 2) return totals;

    for (int i = 1; i < entries.length; i++) {
      final dt = (entries[i].timestampMs - entries[i - 1].timestampMs) / 1000.0;
      if (dt > 300) continue; // ignore gaps > 5 min
      for (final a in kAppliances) {
        if (entries[i - 1].isOn(a)) totals[a] = totals[a]! + dt;
      }
    }
    return totals;
  }

  /// Returns estimated energy in kWh per appliance.
  Map<String, double> computeEnergyKwh(List<HistoryEntry> entries) {
    final runtime = computeRuntimeSeconds(entries);
    return {
      for (final a in kAppliances)
        a: (runtime[a]! / 3600) * (kNominalPower[a]! / 1000)
    };
  }

  /// Returns number of ON→OFF transitions (activations) per appliance.
  Map<String, int> computeActivations(List<HistoryEntry> entries) {
    final counts = {for (final a in kAppliances) a: 0};
    if (entries.length < 2) return counts;

    for (int i = 1; i < entries.length; i++) {
      for (final a in kAppliances) {
        if (!entries[i - 1].isOn(a) && entries[i].isOn(a)) {
          counts[a] = counts[a]! + 1;
        }
      }
    }
    return counts;
  }

  /// Returns average session duration in seconds per appliance.
  Map<String, double> computeAvgSession(List<HistoryEntry> entries) {
    final runtime     = computeRuntimeSeconds(entries);
    final activations = computeActivations(entries);
    return {
      for (final a in kAppliances)
        a: activations[a]! > 0 ? runtime[a]! / activations[a]! : 0.0
    };
  }
}
