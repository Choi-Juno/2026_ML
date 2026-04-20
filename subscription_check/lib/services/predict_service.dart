import 'dart:convert';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:http/http.dart' as http;
import '../config/server_config.dart';
import '../models/subscription.dart';
import '../providers/subscription_provider.dart';

String get _baseUrl {
  if (kIsWeb) return 'http://localhost:5050';
  return 'https://$serverHost';
}

Future<Map<String, ChurnResult>> predictBatch(
    List<Subscription> subscriptions) async {
  final body = subscriptions.map((s) => s.toApiJson()).toList();

  final response = await http.post(
    Uri.parse('$_baseUrl/predict_batch'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode(body),
  );

  if (response.statusCode != 200) {
    throw Exception('서버 응답 오류: ${response.statusCode}');
  }

  final Map<String, dynamic> data = jsonDecode(response.body);
  final results = <String, ChurnResult>{};

  for (final entry in data.entries) {
    final r = entry.value as Map<String, dynamic>;
    results[entry.key] = ChurnResult(
      isChurnCandidate: r['is_churn_candidate'] as bool,
      confidence: (r['confidence'] as num).toDouble(),
      reason: r['reason'] as String,
    );
  }
  return results;
}

Future<bool> checkServerHealth() async {
  try {
    final response = await http
        .get(Uri.parse('$_baseUrl/health'))
        .timeout(const Duration(seconds: 3));
    return response.statusCode == 200;
  } catch (_) {
    return false;
  }
}
