import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'providers/nilm_provider.dart';
import 'screens/home_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
    statusBarColor: Colors.transparent,
    statusBarIconBrightness: Brightness.light,
  ));
  runApp(
    ChangeNotifierProvider(
      create: (_) => NilmProvider(),
      child: const NilmApp(),
    ),
  );
}

class NilmApp extends StatelessWidget {
  const NilmApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'NILM Monitor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF0D0D1A),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF7C3AED),
          secondary: Color(0xFF00E5A0),
          surface: Color(0xFF16162A),
        ),
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}
