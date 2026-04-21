"""
CatBoost 추론 서버 + 피드백 수집 + 재학습 엔드포인트

엔드포인트:
  POST /predict          단건 예측 (DB 저장)
  POST /predict_batch    다건 예측 (DB 저장)
  POST /feedback         실제 결과 피드백 (target 업데이트)
  POST /retrain          즉시 재학습 트리거 (스케줄러 / 관리자용)
  GET  /stats            누적 데이터 통계
  GET  /health           헬스체크
"""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from flask import Flask, request, jsonify
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler

from db import init_db, session_scope, Prediction, AppSubscription
from retrain import (
    feature_engineer, run_retrain,
    CAT_COLS, NUM_COLS, USE_FREQ_MAP, RECENCY_MAP,
)

app = Flask(__name__)
CORS(app)

# ── 전역 변수 ────────────────────────────────────────────────────────────────
model: CatBoostClassifier = None
dataset_stats: dict = {}
current_model_version: str = "initial"

SERVER_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(SERVER_DIR, "model.cbm")
STATS_PATH = os.path.join(SERVER_DIR, "stats.json")

RETRAIN_TOKEN = os.environ.get("RETRAIN_TOKEN")  # 설정 시 /retrain 호출에 필요

# 스케줄 재학습: 매일 RETRAIN_HOUR 시(UTC)에 실행 (기본 18 UTC = KST 새벽 3시)
RETRAIN_HOUR = int(os.environ.get("RETRAIN_HOUR", 18))
RETRAIN_MIN_FEEDBACK = int(os.environ.get("RETRAIN_MIN_FEEDBACK", 10))


# ── 모델 로드 ────────────────────────────────────────────────────────────────
def load_model():
    global model, dataset_stats, current_model_version

    if not (os.path.exists(MODEL_PATH) and os.path.exists(STATS_PATH)):
        raise FileNotFoundError(
            "model.cbm / stats.json 이 없습니다. 먼저 `python retrain.py` 를 실행하세요."
        )

    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)

    with open(STATS_PATH) as f:
        dataset_stats.clear()
        dataset_stats.update(json.load(f))

    # 모델 버전 = 파일 mtime 을 유닉스 초로 단순 표기
    current_model_version = datetime.utcfromtimestamp(
        os.path.getmtime(MODEL_PATH)
    ).strftime("%Y%m%d-%H%M%S")

    print(f"[SERVER] Model loaded version={current_model_version} stats={dataset_stats}")


# ── 이유 문구 ────────────────────────────────────────────────────────────────
def build_reason(row: dict, is_churn: bool) -> str:
    freq    = USE_FREQ_MAP.get(row.get("use_frequency", ""), 0)
    recency = RECENCY_MAP.get(row.get("last_use_recency", ""), 0)

    if not is_churn:
        reasons = []
        if freq >= 3:    reasons.append("이용 빈도 높음")
        if recency >= 3: reasons.append("최근 사용 이력 있음")
        if row.get("perceived_necessity", 0) >= 4: reasons.append("체감 필요도 높음")
        if row.get("would_rebuy", 0) >= 4:          reasons.append("재구독 의향 높음")
        if row.get("cost_burden", 0) <= 2:          reasons.append("비용 부담 낮음")
        return " · ".join(reasons[:3]) if reasons else "전반적으로 유지 가치가 있습니다."

    reasons = []
    if freq <= 2:    reasons.append("이용 빈도 낮음")
    if recency <= 2: reasons.append("오래 미사용")
    emc = max(0, row.get("monthly_cost", 0) - row.get("discount_amount", 0))
    if emc >= dataset_stats.get("high_cost_threshold", 18500):
        reasons.append("실질 비용 높음")
    if row.get("would_rebuy", 0) <= 2:          reasons.append("재구독 의향 낮음")
    if row.get("cost_burden", 0) >= 4:          reasons.append("비용 부담 큼")
    if row.get("replacement_available", 0) == 1: reasons.append("대체 서비스 존재")
    return " · ".join(reasons[:4]) if reasons else "모델 판단: 해지 고려 대상"


# ── 자동 추정 보조 ─────────────────────────────────────────────────────────
def _estimate_cost_burden(monthly_cost: int) -> int:
    if monthly_cost <= 5000:  return 1
    if monthly_cost <= 10000: return 2
    if monthly_cost <= 20000: return 3
    if monthly_cost <= 30000: return 4
    return 5


def _estimate_would_rebuy(freq_score: int, necessity: int) -> int:
    return max(1, min(5, round((freq_score + necessity) / 2)))


# ── 단건 예측 (DB 저장 포함) ───────────────────────────────────────────────
def predict_and_log(raw: dict, session) -> dict:
    monthly_cost        = int(raw.get("monthly_cost", 0))
    use_frequency       = raw.get("use_frequency", "monthly")
    perceived_necessity = int(raw.get("perceived_necessity", 3))
    freq_score          = USE_FREQ_MAP.get(use_frequency, 2)

    row = {
        "subscription_type":     raw.get("subscription_type", "Video"),
        "monthly_cost":          monthly_cost,
        "use_frequency":         use_frequency,
        "last_use_recency":      raw.get("last_use_recency", "7-30d"),
        "perceived_necessity":   perceived_necessity,
        "cost_burden":           int(raw["cost_burden"]) if "cost_burden" in raw else _estimate_cost_burden(monthly_cost),
        "would_rebuy":           int(raw["would_rebuy"]) if "would_rebuy" in raw else _estimate_would_rebuy(freq_score, perceived_necessity),
        "replacement_available": int(raw.get("replacement_available", 0)),
        "billing_cycle":         int(raw.get("billing_cycle", 0)),
        "remaining_months":      float(raw.get("remaining_months", 0)),
        "discount_amount":       int(raw.get("discount_amount", 0)),
    }

    df = feature_engineer(pd.DataFrame([row]), dataset_stats)
    feature_cols = [c for c in CAT_COLS + NUM_COLS if c in df.columns]
    X = df[feature_cols].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype(str)

    pred      = int(model.predict(X).ravel()[0])
    proba     = model.predict_proba(X)[0]
    prob_keep = float(proba[1])

    is_churn   = (pred == 0)
    confidence = (1 - prob_keep) if is_churn else prob_keep

    # DB 저장
    record = Prediction(
        **row,
        predicted_churn=is_churn,
        predicted_confidence=round(confidence, 4),
        model_version=current_model_version,
    )
    session.add(record)
    session.flush()  # id 채번

    return {
        "prediction_id":      record.id,
        "is_churn_candidate": is_churn,
        "confidence":         round(confidence, 4),
        "reason":             build_reason(row, is_churn),
    }


# ── 기기 식별 헬퍼 ────────────────────────────────────────────────────────
def _require_device_id() -> str:
    """X-Device-Id 헤더 필수. 없으면 400."""
    device_id = request.headers.get("X-Device-Id", "").strip()
    if not device_id:
        return ""
    return device_id


# ── 구독 CRUD 엔드포인트 ─────────────────────────────────────────────────
SUBSCRIPTION_WRITABLE_FIELDS = {
    "name", "emoji", "subscription_type", "monthly_cost", "use_frequency",
    "last_use_recency", "perceived_necessity", "cost_burden", "would_rebuy",
    "replacement_available", "is_annual", "remaining_months", "discount_amount",
}


@app.route("/subscriptions", methods=["GET"])
def list_subscriptions():
    device_id = _require_device_id()
    if not device_id:
        return jsonify({"error": "X-Device-Id header required"}), 400

    with session_scope() as session:
        rows = (
            session.query(AppSubscription)
            .filter(AppSubscription.device_id == device_id)
            .order_by(AppSubscription.created_at.asc())
            .all()
        )
        return jsonify([r.to_dict() for r in rows])


@app.route("/subscriptions", methods=["POST"])
def create_subscription():
    device_id = _require_device_id()
    if not device_id:
        return jsonify({"error": "X-Device-Id header required"}), 400

    data = request.get_json(silent=True) or {}
    payload = {k: v for k, v in data.items() if k in SUBSCRIPTION_WRITABLE_FIELDS}

    required = {"name", "subscription_type", "monthly_cost",
                "use_frequency", "last_use_recency", "perceived_necessity"}
    missing = required - payload.keys()
    if missing:
        return jsonify({"error": f"missing fields: {sorted(missing)}"}), 400

    with session_scope() as session:
        row = AppSubscription(device_id=device_id, **payload)
        session.add(row)
        session.flush()
        result = row.to_dict()
    return jsonify(result), 201


@app.route("/subscriptions/<int:sub_id>", methods=["PATCH"])
def update_subscription(sub_id: int):
    device_id = _require_device_id()
    if not device_id:
        return jsonify({"error": "X-Device-Id header required"}), 400

    data = request.get_json(silent=True) or {}
    updates = {k: v for k, v in data.items() if k in SUBSCRIPTION_WRITABLE_FIELDS}
    if not updates:
        return jsonify({"error": "no updatable fields"}), 400

    with session_scope() as session:
        row = (
            session.query(AppSubscription)
            .filter(AppSubscription.id == sub_id,
                    AppSubscription.device_id == device_id)
            .one_or_none()
        )
        if row is None:
            return jsonify({"error": "subscription not found"}), 404
        for k, v in updates.items():
            setattr(row, k, v)
        session.flush()
        result = row.to_dict()
    return jsonify(result)


@app.route("/subscriptions/<int:sub_id>", methods=["DELETE"])
def delete_subscription(sub_id: int):
    device_id = _require_device_id()
    if not device_id:
        return jsonify({"error": "X-Device-Id header required"}), 400

    with session_scope() as session:
        row = (
            session.query(AppSubscription)
            .filter(AppSubscription.id == sub_id,
                    AppSubscription.device_id == device_id)
            .one_or_none()
        )
        if row is None:
            return jsonify({"error": "subscription not found"}), 404
        session.delete(row)
    return jsonify({"ok": True})


# ── 엔드포인트 ────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    with session_scope() as session:
        result = predict_and_log(data, session)
    return jsonify(result)


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    items = request.get_json(silent=True)
    if not isinstance(items, list):
        return jsonify({"error": "Expected JSON array"}), 400

    results = {}
    with session_scope() as session:
        for item in items:
            sid = item.get("id", "")
            results[sid] = predict_and_log(item, session)
    return jsonify(results)


@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Body: { "prediction_id": int, "actual_kept": bool }
    actual_kept=true  → target=1 (유지)
    actual_kept=false → target=0 (해지)
    """
    data = request.get_json(silent=True) or {}
    pid = data.get("prediction_id")
    if pid is None or "actual_kept" not in data:
        return jsonify({"error": "prediction_id, actual_kept required"}), 400

    target = 1 if bool(data["actual_kept"]) else 0

    with session_scope() as session:
        row = session.get(Prediction, int(pid))
        if row is None:
            return jsonify({"error": "prediction not found"}), 404
        row.actual_target = target
        row.feedback_at   = datetime.utcnow()

    return jsonify({"ok": True, "prediction_id": int(pid), "actual_target": target})


@app.route("/retrain", methods=["POST"])
def retrain_endpoint():
    """스케줄러 또는 관리자가 호출. RETRAIN_TOKEN 이 설정되어 있으면 헤더로 검증."""
    if RETRAIN_TOKEN:
        token = request.headers.get("X-Retrain-Token", "")
        if token != RETRAIN_TOKEN:
            return jsonify({"error": "unauthorized"}), 401

    result = run_retrain()
    load_model()  # 새 모델 즉시 반영
    return jsonify({"ok": True, **result, "loaded_version": current_model_version})


@app.route("/stats", methods=["GET"])
def stats():
    with session_scope() as session:
        total     = session.query(Prediction).count()
        labeled   = session.query(Prediction).filter(Prediction.actual_target.isnot(None)).count()
        kept      = session.query(Prediction).filter(Prediction.actual_target == 1).count()
        churned   = session.query(Prediction).filter(Prediction.actual_target == 0).count()

    return jsonify({
        "total_predictions": total,
        "labeled_samples":   labeled,
        "kept_count":        kept,
        "churned_count":     churned,
        "model_version":     current_model_version,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":        "ok",
        "model_loaded":  model is not None,
        "model_version": current_model_version,
    })


# ── 스케줄러 ──────────────────────────────────────────────────────────────
def scheduled_retrain_job():
    """백그라운드 스레드에서 실행. 새 피드백이 충분할 때만 재학습."""
    try:
        with session_scope() as session:
            labeled = (
                session.query(Prediction)
                .filter(Prediction.actual_target.isnot(None))
                .count()
            )
        if labeled < RETRAIN_MIN_FEEDBACK:
            print(f"[SCHEDULER] Skip retrain (labeled={labeled} < min={RETRAIN_MIN_FEEDBACK})")
            return
        print(f"[SCHEDULER] Retraining with {labeled} labeled samples...")
        run_retrain()
        load_model()
        print(f"[SCHEDULER] Retrain done. version={current_model_version}")
    except Exception as e:
        print(f"[SCHEDULER] Retrain failed: {e}")


# ── 부트스트랩 ────────────────────────────────────────────────────────────
def bootstrap():
    """서버 시작 전 필수 초기화."""
    init_db()
    load_model()

    scheduler = BackgroundScheduler(daemon=True, timezone="UTC")
    scheduler.add_job(
        scheduled_retrain_job,
        trigger="cron",
        hour=RETRAIN_HOUR,
        minute=0,
        id="daily_retrain",
    )
    scheduler.start()
    print(f"[SCHEDULER] Daily retrain scheduled at {RETRAIN_HOUR}:00 UTC")


bootstrap()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"[SERVER] Starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
