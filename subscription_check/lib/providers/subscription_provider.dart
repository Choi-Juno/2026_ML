import 'dart:async';

import 'package:flutter/foundation.dart';
import '../models/subscription.dart';
import '../services/predict_service.dart';

class ChurnResult {
  final int predictionId;
  final bool isChurnCandidate;
  final double confidence;
  final String reason;
  final bool? userFeedbackKept;

  ChurnResult({
    required this.predictionId,
    required this.isChurnCandidate,
    required this.confidence,
    required this.reason,
    this.userFeedbackKept,
  });

  ChurnResult copyWith({bool? userFeedbackKept}) => ChurnResult(
        predictionId: predictionId,
        isChurnCandidate: isChurnCandidate,
        confidence: confidence,
        reason: reason,
        userFeedbackKept: userFeedbackKept ?? this.userFeedbackKept,
      );
}

const _analyzeDebounce = Duration(milliseconds: 600);

class SubscriptionProvider extends ChangeNotifier {
  final List<Subscription> _items = [];
  Map<String, ChurnResult> _results = {};

  bool _isLoading = true;
  bool _isAnalyzing = false;
  String? _errorMessage;
  Timer? _debounce;

  List<Subscription> get items => List.unmodifiable(_items);
  Map<String, ChurnResult> get results => Map.unmodifiable(_results);
  bool get isLoading => _isLoading;
  bool get isAnalyzing => _isAnalyzing;
  String? get errorMessage => _errorMessage;

  int get totalMonthlyCost =>
      _items.fold(0, (sum, s) => sum + s.effectiveMonthlyCost);

  int get saveableCost {
    var sum = 0;
    for (final s in _items) {
      final r = _results[s.id];
      if (r != null && r.isChurnCandidate) {
        sum += s.effectiveMonthlyCost;
      }
    }
    return sum;
  }

  /// 서버 ID 로 관리되므로 더미 nextId는 필요 없음. 호환용으로 남겨둠.
  String get nextId => DateTime.now().microsecondsSinceEpoch.toString();

  Future<void> loadFromServer() async {
    _isLoading = true;
    _errorMessage = null;
    notifyListeners();

    try {
      final fetched = await fetchSubscriptions();
      _items
        ..clear()
        ..addAll(fetched);
      _results.clear();
    } catch (e) {
      _errorMessage = '구독 목록을 불러올 수 없습니다.';
    }

    _isLoading = false;
    notifyListeners();

    if (_items.isNotEmpty) {
      _scheduleAnalyze();
    }
  }

  Future<void> addSubscription(Subscription s) async {
    try {
      final created = await createSubscription(s);
      _items.add(created);
      notifyListeners();
      _scheduleAnalyze();
    } catch (e) {
      _errorMessage = '구독 추가에 실패했습니다.';
      notifyListeners();
    }
  }

  Future<void> removeSubscription(String id) async {
    final idx = _items.indexWhere((s) => s.id == id);
    if (idx < 0) return;
    final removed = _items.removeAt(idx);
    _results.remove(id);
    notifyListeners();

    try {
      await deleteSubscription(id);
      _scheduleAnalyze();
    } catch (e) {
      _items.insert(idx, removed); // rollback
      _errorMessage = '구독 삭제에 실패했습니다.';
      notifyListeners();
    }
  }

  Future<void> updateSubscription(Subscription updated) async {
    final idx = _items.indexWhere((s) => s.id == updated.id);
    if (idx < 0) return;
    final previous = _items[idx];
    _items[idx] = updated;
    notifyListeners();

    try {
      final synced = await patchSubscription(updated);
      _items[idx] = synced;
      notifyListeners();
      _scheduleAnalyze();
    } catch (e) {
      _items[idx] = previous; // rollback
      _errorMessage = '구독 수정에 실패했습니다.';
      notifyListeners();
    }
  }

  void _scheduleAnalyze() {
    _debounce?.cancel();
    if (_items.isEmpty) return;
    _debounce = Timer(_analyzeDebounce, analyzeAll);
  }

  Future<void> analyzeAll() async {
    if (_items.isEmpty) return;
    _isAnalyzing = true;
    _errorMessage = null;
    notifyListeners();

    try {
      _results = await predictBatch(_items);
    } catch (e) {
      _errorMessage = '추론 서버에 연결할 수 없습니다.';
    }

    _isAnalyzing = false;
    notifyListeners();
  }

  Future<void> submitChurnFeedback({
    required String subscriptionId,
    required bool actualKept,
  }) async {
    final result = _results[subscriptionId];
    if (result == null) return;

    try {
      await submitFeedback(
        predictionId: result.predictionId,
        actualKept: actualKept,
      );
      _results[subscriptionId] = result.copyWith(userFeedbackKept: actualKept);
      notifyListeners();
    } catch (e) {
      _errorMessage = '피드백 전송에 실패했습니다.';
      notifyListeners();
    }
  }

  @override
  void dispose() {
    _debounce?.cancel();
    super.dispose();
  }
}
