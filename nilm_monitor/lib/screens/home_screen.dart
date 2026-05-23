import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/nilm_provider.dart';
import '../models/appliance_state.dart';
import '../widgets/appliance_card.dart';
import 'settings_screen.dart';
import 'history_screen.dart';

const _bgColor      = Color(0xFF0D0D1A);
const _surfaceColor = Color(0xFF16162A);
const _surface2     = Color(0xFF1F1F3A);
const _borderColor  = Color(0xFF2A2A4A);
const _accentColor  = Color(0xFF7C3AED);
const _accent2      = Color(0xFFA855F7);
const _onColor      = Color(0xFF00E5A0);
const _offColor     = Color(0xFF3A3A5C);
const _lowColor     = Color(0xFFF59E0B);
const _textColor    = Color(0xFFE2E2F0);
const _textDim      = Color(0xFF8888AA);
const _startColor   = Color(0xFF059669);
const _startColor2  = Color(0xFF10B981);
const _stopColor    = Color(0xFFDC2626);
const _stopColor2   = Color(0xFFEF4444);

const _appliances = ['kettle', 'microwave', 'fridge', 'tv'];

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen>
    with SingleTickerProviderStateMixin {
  late final AnimationController _pulseCtrl = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 1200),
  )..repeat(reverse: true);

  @override
  void dispose() {
    _pulseCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _bgColor,
      body: SafeArea(
        child: Consumer<NilmProvider>(
          builder: (context, p, _) {
            return CustomScrollView(
              slivers: [
                _buildAppBar(context, p),
                SliverPadding(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  sliver: SliverList(
                    delegate: SliverChildListDelegate([
                      const SizedBox(height: 16),
                      _buildStreamControl(p),
                      const SizedBox(height: 12),
                      _buildHouseSelector(p),
                      const SizedBox(height: 12),
                      _buildStatsRow(p),
                      const SizedBox(height: 10),
                      _buildWindowProgress(p),
                      if (p.error != null) ...[
                        const SizedBox(height: 10),
                        _buildError(p.error!),
                      ],
                      const SizedBox(height: 14),
                      _buildApplianceGrid(p),
                      const SizedBox(height: 20),
                      _buildHistory(p),
                      const SizedBox(height: 24),
                    ]),
                  ),
                ),
              ],
            );
          },
        ),
      ),
    );
  }

  // ── App Bar ──────────────────────────────────────────────────────────
  SliverAppBar _buildAppBar(BuildContext context, NilmProvider p) {
    return SliverAppBar(
      backgroundColor: _surfaceColor,
      pinned: true,
      elevation: 0,
      titleSpacing: 16,
      title: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: _accentColor.withValues(alpha: 0.2),
              borderRadius: BorderRadius.circular(10),
            ),
            child: const Icon(Icons.bolt, color: _accent2, size: 20),
          ),
          const SizedBox(width: 10),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('NILM Monitor',
                  style: TextStyle(
                      color: _textColor, fontWeight: FontWeight.w700, fontSize: 16)),
              Text('${p.ip}:${p.port}',
                  style: const TextStyle(color: _textDim, fontSize: 11)),
            ],
          ),
        ],
      ),
      actions: [
        IconButton(
          icon: const Icon(Icons.bar_chart_rounded, color: _textDim),
          tooltip: 'Historique',
          onPressed: () => Navigator.push(context,
              MaterialPageRoute(builder: (_) => const HistoryScreen())),
        ),
        IconButton(
          icon: const Icon(Icons.settings_outlined, color: _textDim),
          onPressed: () => Navigator.push(context,
              MaterialPageRoute(builder: (_) => const SettingsScreen())),
        ),
      ],
    );
  }

  // ── Stream Control ───────────────────────────────────────────────────
  Widget _buildStreamControl(NilmProvider p) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: _surfaceColor,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: _borderColor),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('STREAMING DEPUIS LE DATASET',
              style: TextStyle(
                  fontSize: 10,
                  fontWeight: FontWeight.w700,
                  color: _textDim,
                  letterSpacing: 1)),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: _gradientButton(
                  label: 'Démarrer',
                  icon: Icons.play_arrow_rounded,
                  colors: [_startColor, _startColor2],
                  enabled: !p.isStreaming,
                  onTap: p.startStream,
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: _gradientButton(
                  label: 'Arrêter',
                  icon: Icons.stop_rounded,
                  colors: [_stopColor, _stopColor2],
                  enabled: p.isStreaming,
                  onTap: p.stopStream,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              // Animated pulse dot
              AnimatedBuilder(
                animation: _pulseCtrl,
                builder: (_, __) => Container(
                  width: 8,
                  height: 8,
                  decoration: BoxDecoration(
                    color: p.isStreaming
                        ? _onColor.withValues(
                            alpha: 0.35 + 0.65 * _pulseCtrl.value)
                        : _offColor,
                    shape: BoxShape.circle,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  p.isStreaming
                      ? '${p.windowsCompleted} fenêtre${p.windowsCompleted != 1 ? "s" : ""} inférée${p.windowsCompleted != 1 ? "s" : ""} · '
                        'échantillon ${p.samplesBuffered}/${p.windowSize} en cours'
                      : p.windowsCompleted > 0
                          ? 'Stream terminé — ${p.windowsCompleted} fenêtre${p.windowsCompleted != 1 ? "s" : ""} traitée${p.windowsCompleted != 1 ? "s" : ""}'
                          : 'En attente de démarrage…',
                  style: const TextStyle(fontSize: 12, color: _textDim),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _gradientButton({
    required String label,
    required IconData icon,
    required List<Color> colors,
    required bool enabled,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: enabled ? onTap : null,
      child: AnimatedOpacity(
        opacity: enabled ? 1.0 : 0.35,
        duration: const Duration(milliseconds: 200),
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 13),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              colors: enabled ? colors : [_offColor, _offColor],
              begin: Alignment.centerLeft,
              end: Alignment.centerRight,
            ),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(icon, color: Colors.white, size: 18),
              const SizedBox(width: 6),
              Text(label,
                  style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w700,
                      fontSize: 14)),
            ],
          ),
        ),
      ),
    );
  }

  // ── House Selector ───────────────────────────────────────────────────
  Widget _buildHouseSelector(NilmProvider p) {
    return Row(
      children: [
        ...[3, 9].map((h) {
          final active = p.house == h;
          return Padding(
            padding: EdgeInsets.only(right: h == 3 ? 8 : 0),
            child: GestureDetector(
              onTap: () => p.setHouse(h),
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 200),
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 9),
                decoration: BoxDecoration(
                  color: active ? _accentColor : _surfaceColor,
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: active ? _accent2 : _borderColor),
                ),
                child: Text(
                  h == 3 ? '🏡  Maison 3' : '🏘️  Maison 9',
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                    color: active ? Colors.white : _textDim,
                    letterSpacing: 0.5,
                  ),
                ),
              ),
            ),
          );
        }),
        const Spacer(),
        GestureDetector(
          onTap: () => p.reset(),
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 9),
            decoration: BoxDecoration(
              color: _surfaceColor,
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: _borderColor),
            ),
            child: const Row(
              children: [
                Icon(Icons.refresh, color: _textDim, size: 14),
                SizedBox(width: 4),
                Text('↺  Réinitialiser',
                    style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w700,
                        color: _textDim,
                        letterSpacing: 0.3)),
              ],
            ),
          ),
        ),
      ],
    );
  }

  // ── Stats Row ────────────────────────────────────────────────────────
  Widget _buildStatsRow(NilmProvider p) {
    final agg = p.meanAggregate;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: _surface2,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: _borderColor),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _statItem('AGGREGATE',
              agg != null ? '${agg.toStringAsFixed(1)} W' : '— W'),
          _divider(),
          _statItem('FENÊTRES', '${p.windowsCompleted}'),
          _divider(),
          _statusItem(p),
        ],
      ),
    );
  }

  Widget _statItem(String label, String value) => Column(
        children: [
          Text(label,
              style: const TextStyle(
                  fontSize: 9, color: _textDim, letterSpacing: 0.8, fontWeight: FontWeight.w600)),
          const SizedBox(height: 4),
          Text(value,
              style: const TextStyle(
                  fontSize: 13, color: _textColor, fontWeight: FontWeight.w700)),
        ],
      );

  Widget _statusItem(NilmProvider p) => Column(
        children: [
          const Text('STATUS',
              style: TextStyle(
                  fontSize: 9, color: _textDim, letterSpacing: 0.8, fontWeight: FontWeight.w600)),
          const SizedBox(height: 4),
          Row(
            children: [
              Container(
                width: 7,
                height: 7,
                decoration: BoxDecoration(
                  color: _statusColor(p),
                  shape: BoxShape.circle,
                ),
              ),
              const SizedBox(width: 5),
              Text(
                _statusLabel(p),
                style: TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w700,
                    color: _statusColor(p)),
              ),
            ],
          ),
        ],
      );

  Color _statusColor(NilmProvider p) {
    if (!p.isStreaming && p.lastResult == null) return _offColor;
    if (p.isReady) return _onColor;
    if (p.isStreaming) return _lowColor;
    return _offColor;
  }

  String _statusLabel(NilmProvider p) {
    if (!p.isStreaming && p.lastResult == null) return 'Inactif';
    if (p.isReady) return 'Prêt';
    if (p.isStreaming) return 'Collecte';
    return 'Inactif';
  }

  Widget _divider() => Container(width: 1, height: 28, color: _borderColor);

  // ── Window Progress ──────────────────────────────────────────────────
  Widget _buildWindowProgress(NilmProvider p) {
    final buffered = p.samplesBuffered;
    final total    = p.windowSize;
    final fraction = total > 0 ? (buffered / total).clamp(0.0, 1.0) : 0.0;
    final ready    = p.isReady;
    final barColor = ready ? _onColor : _accentColor;

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: _surfaceColor,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: _borderColor),
      ),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                ready
                    ? 'Fenêtre #${p.windowsCompleted} complète'
                    : p.isStreaming
                        ? 'Collecte · ${total - buffered} échantillon${total - buffered > 1 ? "s" : ""} restant${total - buffered > 1 ? "s" : ""}'
                        : 'Démarrez le stream…',
                style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                    color: ready ? _onColor : _textDim),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: barColor.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: barColor.withValues(alpha: 0.4)),
                ),
                child: Text(
                  '$buffered / $total',
                  style: TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                      color: barColor),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: TweenAnimationBuilder<double>(
              tween: Tween<double>(begin: 0, end: fraction),
              duration: const Duration(milliseconds: 400),
              builder: (_, value, __) => LinearProgressIndicator(
                value: value,
                minHeight: 6,
                backgroundColor: _surface2,
                valueColor: AlwaysStoppedAnimation<Color>(barColor),
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ── Error ────────────────────────────────────────────────────────────
  Widget _buildError(String msg) => Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: Colors.red.withValues(alpha: 0.1),
          borderRadius: BorderRadius.circular(10),
          border: Border.all(color: Colors.red.withValues(alpha: 0.4)),
        ),
        child: Row(
          children: [
            const Icon(Icons.error_outline, color: Colors.redAccent, size: 16),
            const SizedBox(width: 8),
            Expanded(
              child: Text(msg,
                  style: const TextStyle(color: Colors.redAccent, fontSize: 13)),
            ),
          ],
        ),
      );

  // ── Appliance Grid ───────────────────────────────────────────────────
  Widget _buildApplianceGrid(NilmProvider p) {
    final apps = p.lastResult?.appliances;
    return GridView.count(
      crossAxisCount: 2,
      crossAxisSpacing: 12,
      mainAxisSpacing: 12,
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      childAspectRatio: 0.85,
      children: _appliances.map((name) {
        return ApplianceCard(name: name, info: apps?[name]);
      }).toList(),
    );
  }

  // ── History ──────────────────────────────────────────────────────────
  Widget _buildHistory(NilmProvider p) {
    if (p.history.isEmpty) return const SizedBox.shrink();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            const Text('HISTORIQUE DES FENÊTRES',
                style: TextStyle(
                    fontSize: 10,
                    fontWeight: FontWeight.w700,
                    color: _textDim,
                    letterSpacing: 1)),
            Text('${p.history.length} dernières',
                style: const TextStyle(fontSize: 10, color: _textDim)),
          ],
        ),
        const SizedBox(height: 8),
        ...p.history.map((r) => _historyItem(r)),
      ],
    );
  }

  Widget _historyItem(WindowResult r) {
    final tStart = r.tStart ?? '';
    final tEnd   = r.tEnd   ?? '';
    final tS     = tStart.length >= 19 ? tStart.substring(11, 19) : tStart;
    final tE     = tEnd.length   >= 19 ? tEnd.substring(11, 19)   : tEnd;
    final date   = tStart.length >= 10 ? tStart.substring(0, 10)  : '';
    final apps   = r.appliances ?? {};
    final power  = r.meanAggregate ?? 0.0;

    return Container(
      margin: const EdgeInsets.only(bottom: 6),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: _surfaceColor,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: _borderColor),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text('Fenêtre #${r.windowIndex ?? "—"}',
                  style: const TextStyle(
                      fontSize: 12, fontWeight: FontWeight.w700, color: _accent2)),
              const Spacer(),
              Text('${power.round()} W',
                  style: const TextStyle(
                      fontSize: 13, fontWeight: FontWeight.w700, color: _textColor)),
            ],
          ),
          const SizedBox(height: 4),
          Row(
            children: [
              Text(date.isNotEmpty ? '$date  $tS→$tE' : '$tS→$tE',
                  style: const TextStyle(fontSize: 10, color: _textDim)),
              const Spacer(),
              ...apps.entries.map((e) {
                final col = switch (e.value.state) {
                  'ON' || 'HIGH' => _onColor,
                  'LOW'          => _lowColor,
                  _              => _offColor,
                };
                return Padding(
                  padding: const EdgeInsets.only(left: 5),
                  child: _stateTag(e.key, e.value.state, col),
                );
              }),
            ],
          ),
        ],
      ),
    );
  }

  Widget _stateTag(String name, String state, Color col) {
    final icons = {'kettle': '🫖', 'microwave': '📟', 'fridge': '❄️', 'tv': '📺'};
    final isOff = state == 'OFF';
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: col.withValues(alpha: isOff ? 0.0 : 0.15),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: col.withValues(alpha: isOff ? 0.2 : 0.4)),
      ),
      child: Text(
        '${icons[name] ?? ""} $state',
        style: TextStyle(
            fontSize: 10,
            fontWeight: FontWeight.w600,
            color: isOff ? _textDim : col),
      ),
    );
  }
}
