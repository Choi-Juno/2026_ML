enum UseFrequency { rare, monthly, weekly, frequent }

enum LastUseRecency { over30d, between7and30d, between1and7d, under1d }

class Subscription {
  final String id;
  final String name;
  final String type;
  final int monthlyCost;
  final UseFrequency useFrequency;
  final LastUseRecency lastUseRecency;
  final int perceivedNecessity;
  final int? costBurden;
  final int? wouldRebuy;
  final bool replacementAvailable;
  final bool isAnnual;
  final double remainingMonths;
  final int discountAmount;
  final String? emoji;

  int get effectiveMonthlyCost =>
      (monthlyCost - discountAmount).clamp(0, monthlyCost);

  String get useFrequencyApiValue {
    switch (useFrequency) {
      case UseFrequency.rare:
        return 'rare';
      case UseFrequency.monthly:
        return 'monthly';
      case UseFrequency.weekly:
        return 'weekly';
      case UseFrequency.frequent:
        return 'frequent';
    }
  }

  String get lastUseRecencyApiValue {
    switch (lastUseRecency) {
      case LastUseRecency.over30d:
        return '>30d';
      case LastUseRecency.between7and30d:
        return '7-30d';
      case LastUseRecency.between1and7d:
        return '1-7d';
      case LastUseRecency.under1d:
        return '<1d';
    }
  }

  Map<String, dynamic> toApiJson() {
    final map = <String, dynamic>{
      'id': id,
      'subscription_type': type,
      'monthly_cost': monthlyCost,
      'use_frequency': useFrequencyApiValue,
      'last_use_recency': lastUseRecencyApiValue,
      'perceived_necessity': perceivedNecessity,
      'replacement_available': replacementAvailable ? 1 : 0,
      'billing_cycle': isAnnual ? 1 : 0,
      'remaining_months': remainingMonths,
      'discount_amount': discountAmount,
    };
    if (costBurden != null) map['cost_burden'] = costBurden;
    if (wouldRebuy != null) map['would_rebuy'] = wouldRebuy;
    return map;
  }

  /// POST /subscriptions, PATCH /subscriptions/:id 에 보내는 payload.
  Map<String, dynamic> toPersistenceJson() => {
        'name': name,
        'emoji': emoji,
        'subscription_type': type,
        'monthly_cost': monthlyCost,
        'use_frequency': useFrequencyApiValue,
        'last_use_recency': lastUseRecencyApiValue,
        'perceived_necessity': perceivedNecessity,
        'cost_burden': costBurden,
        'would_rebuy': wouldRebuy,
        'replacement_available': replacementAvailable,
        'is_annual': isAnnual,
        'remaining_months': remainingMonths,
        'discount_amount': discountAmount,
      };

  static Subscription fromServerJson(Map<String, dynamic> json) {
    return Subscription(
      id: json['id'].toString(),
      name: json['name'] as String,
      type: json['subscription_type'] as String,
      monthlyCost: (json['monthly_cost'] as num).toInt(),
      useFrequency: _parseFrequency(json['use_frequency'] as String),
      lastUseRecency: _parseRecency(json['last_use_recency'] as String),
      perceivedNecessity: (json['perceived_necessity'] as num).toInt(),
      costBurden: (json['cost_burden'] as num?)?.toInt(),
      wouldRebuy: (json['would_rebuy'] as num?)?.toInt(),
      replacementAvailable: json['replacement_available'] as bool? ?? false,
      isAnnual: json['is_annual'] as bool? ?? false,
      remainingMonths: (json['remaining_months'] as num?)?.toDouble() ?? 0.0,
      discountAmount: (json['discount_amount'] as num?)?.toInt() ?? 0,
      emoji: json['emoji'] as String?,
    );
  }

  static UseFrequency _parseFrequency(String value) {
    switch (value) {
      case 'rare':
        return UseFrequency.rare;
      case 'weekly':
        return UseFrequency.weekly;
      case 'frequent':
        return UseFrequency.frequent;
      case 'monthly':
      default:
        return UseFrequency.monthly;
    }
  }

  static LastUseRecency _parseRecency(String value) {
    switch (value) {
      case '>30d':
        return LastUseRecency.over30d;
      case '1-7d':
        return LastUseRecency.between1and7d;
      case '<1d':
        return LastUseRecency.under1d;
      case '7-30d':
      default:
        return LastUseRecency.between7and30d;
    }
  }

  Subscription copyWith({
    String? id,
    String? name,
    String? type,
    int? monthlyCost,
    UseFrequency? useFrequency,
    LastUseRecency? lastUseRecency,
    int? perceivedNecessity,
    int? costBurden,
    int? wouldRebuy,
    bool? replacementAvailable,
    bool? isAnnual,
    double? remainingMonths,
    int? discountAmount,
    String? emoji,
  }) {
    return Subscription(
      id: id ?? this.id,
      name: name ?? this.name,
      type: type ?? this.type,
      monthlyCost: monthlyCost ?? this.monthlyCost,
      useFrequency: useFrequency ?? this.useFrequency,
      lastUseRecency: lastUseRecency ?? this.lastUseRecency,
      perceivedNecessity: perceivedNecessity ?? this.perceivedNecessity,
      costBurden: costBurden ?? this.costBurden,
      wouldRebuy: wouldRebuy ?? this.wouldRebuy,
      replacementAvailable: replacementAvailable ?? this.replacementAvailable,
      isAnnual: isAnnual ?? this.isAnnual,
      remainingMonths: remainingMonths ?? this.remainingMonths,
      discountAmount: discountAmount ?? this.discountAmount,
      emoji: emoji ?? this.emoji,
    );
  }

  Subscription({
    required this.id,
    required this.name,
    required this.type,
    required this.monthlyCost,
    required this.useFrequency,
    required this.lastUseRecency,
    required this.perceivedNecessity,
    this.costBurden,
    this.wouldRebuy,
    required this.replacementAvailable,
    this.isAnnual = false,
    this.remainingMonths = 0,
    this.discountAmount = 0,
    this.emoji,
  });
}

String freqShortLabel(UseFrequency f) {
  switch (f) {
    case UseFrequency.rare:
      return '드물게';
    case UseFrequency.monthly:
      return '월 1~2회';
    case UseFrequency.weekly:
      return '주 1~2회';
    case UseFrequency.frequent:
      return '거의 매일';
  }
}

String recencyShortLabel(LastUseRecency r) {
  switch (r) {
    case LastUseRecency.over30d:
      return '30일+';
    case LastUseRecency.between7and30d:
      return '7-30일';
    case LastUseRecency.between1and7d:
      return '1-7일';
    case LastUseRecency.under1d:
      return '1일 이내';
  }
}
