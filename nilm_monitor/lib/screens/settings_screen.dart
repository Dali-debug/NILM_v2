import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/nilm_provider.dart';

const _bgColor      = Color(0xFF0D0D1A);
const _surfaceColor = Color(0xFF16162A);
const _surface2     = Color(0xFF1F1F3A);
const _borderColor  = Color(0xFF2A2A4A);
const _accentColor  = Color(0xFF7C3AED);
const _accent2      = Color(0xFFA855F7);
const _onColor      = Color(0xFF00E5A0);
const _textColor    = Color(0xFFE2E2F0);
const _textDim      = Color(0xFF8888AA);

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  late TextEditingController _ipCtrl;
  late TextEditingController _portCtrl;
  int _selectedHouse = 3;
  String? _testResult;
  bool _testing = false;

  @override
  void initState() {
    super.initState();
    final p = context.read<NilmProvider>();
    _ipCtrl        = TextEditingController(text: p.ip);
    _portCtrl      = TextEditingController(text: p.port.toString());
    _selectedHouse = p.house;
  }

  @override
  void dispose() {
    _ipCtrl.dispose();
    _portCtrl.dispose();
    super.dispose();
  }

  Future<void> _testConnection() async {
    setState(() { _testing = true; _testResult = null; });
    await context.read<NilmProvider>().saveSettings(
      _ipCtrl.text.trim(),
      int.tryParse(_portCtrl.text.trim()) ?? 5001,
      _selectedHouse,
    );
    final msg = await context.read<NilmProvider>().testConnection();
    setState(() { _testing = false; _testResult = msg; });
  }

  Future<void> _save() async {
    await context.read<NilmProvider>().saveSettings(
      _ipCtrl.text.trim(),
      int.tryParse(_portCtrl.text.trim()) ?? 5001,
      _selectedHouse,
    );
    if (mounted) Navigator.pop(context);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _bgColor,
      appBar: AppBar(
        backgroundColor: _surfaceColor,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new, color: _textColor, size: 18),
          onPressed: () => Navigator.pop(context),
        ),
        title: const Text('Settings',
            style: TextStyle(color: _textColor, fontWeight: FontWeight.w700, fontSize: 18)),
      ),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          _sectionLabel('RASPBERRY PI CONNECTION'),
          const SizedBox(height: 10),

          // IP
          _fieldLabel('IP Address'),
          const SizedBox(height: 6),
          _inputField(
            controller: _ipCtrl,
            hint: '192.168.1.42',
            keyboard: TextInputType.number,
          ),
          const SizedBox(height: 14),

          // Port
          _fieldLabel('Port'),
          const SizedBox(height: 6),
          _inputField(
            controller: _portCtrl,
            hint: '5001',
            keyboard: TextInputType.number,
          ),
          const SizedBox(height: 20),

          // Test connection
          GestureDetector(
            onTap: _testing ? null : _testConnection,
            child: Container(
              padding: const EdgeInsets.symmetric(vertical: 14),
              decoration: BoxDecoration(
                border: Border.all(color: _borderColor),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Center(
                child: _testing
                    ? const SizedBox(
                        width: 18, height: 18,
                        child: CircularProgressIndicator(
                          strokeWidth: 2, color: _accent2))
                    : const Text('Test Connection',
                        style: TextStyle(color: _textDim, fontWeight: FontWeight.w600)),
              ),
            ),
          ),

          if (_testResult != null) ...[
            const SizedBox(height: 10),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: _testResult!.startsWith('Connecté')
                    ? _onColor.withOpacity(0.1)
                    : Colors.red.withOpacity(0.1),
                borderRadius: BorderRadius.circular(10),
                border: Border.all(
                  color: _testResult!.startsWith('Connecté')
                      ? _onColor.withOpacity(0.4)
                      : Colors.red.withOpacity(0.4),
                ),
              ),
              child: Text(
                _testResult!,
                style: TextStyle(
                  fontSize: 13,
                  color: _testResult!.startsWith('Connecté') ? _onColor : Colors.redAccent,
                ),
              ),
            ),
          ],

          const SizedBox(height: 28),
          _sectionLabel('DEFAULT HOUSE'),
          const SizedBox(height: 10),

          Row(
            children: [3, 9].map((h) {
              final active = _selectedHouse == h;
              return Expanded(
                child: Padding(
                  padding: EdgeInsets.only(right: h == 3 ? 8 : 0),
                  child: GestureDetector(
                    onTap: () => setState(() => _selectedHouse = h),
                    child: AnimatedContainer(
                      duration: const Duration(milliseconds: 200),
                      padding: const EdgeInsets.symmetric(vertical: 12),
                      decoration: BoxDecoration(
                        color: active ? _accentColor : _surfaceColor,
                        borderRadius: BorderRadius.circular(10),
                        border: Border.all(
                          color: active ? _accent2 : _borderColor,
                        ),
                      ),
                      child: Center(
                        child: Text(
                          'House $h',
                          style: TextStyle(
                            fontWeight: FontWeight.w700,
                            color: active ? Colors.white : _textDim,
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
              );
            }).toList(),
          ),

          const SizedBox(height: 32),

          // Save button
          GestureDetector(
            onTap: _save,
            child: Container(
              padding: const EdgeInsets.symmetric(vertical: 16),
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  colors: [_accentColor, _accent2],
                  begin: Alignment.centerLeft,
                  end: Alignment.centerRight,
                ),
                borderRadius: BorderRadius.circular(14),
              ),
              child: const Center(
                child: Text('Save Settings',
                    style: TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.w700,
                        fontSize: 16)),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _sectionLabel(String text) => Text(
        text,
        style: const TextStyle(
          fontSize: 11,
          fontWeight: FontWeight.w700,
          color: _textDim,
          letterSpacing: 1,
        ),
      );

  Widget _fieldLabel(String text) => Text(
        text,
        style: const TextStyle(fontSize: 13, color: _textDim),
      );

  Widget _inputField({
    required TextEditingController controller,
    required String hint,
    TextInputType keyboard = TextInputType.text,
  }) =>
      TextField(
        controller: controller,
        keyboardType: keyboard,
        style: const TextStyle(color: _textColor, fontWeight: FontWeight.w600),
        decoration: InputDecoration(
          hintText: hint,
          hintStyle: const TextStyle(color: _textDim),
          filled: true,
          fillColor: _surface2,
          contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(color: _borderColor),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(color: _borderColor),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(color: _accent2),
          ),
        ),
      );
}
