import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def dcg_at_k(relevance: np.ndarray, k: int = 10) -> float:
    values = np.asarray(relevance, dtype=float)[:k]
    discounts = np.log2(np.arange(2, len(values) + 2))
    return float(np.sum((2**values - 1) / discounts))


def ndcg_at_k(relevance: np.ndarray, k: int = 10) -> float:
    ideal = np.sort(np.asarray(relevance, dtype=float))[::-1]
    best = dcg_at_k(ideal, k)
    return 0.0 if best == 0 else dcg_at_k(relevance, k) / best


def mrr_at_k(relevance: np.ndarray, k: int = 10) -> float:
    hits = np.flatnonzero(np.asarray(relevance)[:k] > 0)
    return 0.0 if len(hits) == 0 else 1.0 / float(hits[0] + 1)


def success_at_k(relevance: np.ndarray, k: int = 3) -> float:
    return float(np.any(np.asarray(relevance)[:k] > 0))


def make_baseline(rng: np.random.Generator) -> np.ndarray:
    ranks = np.arange(10)
    hit_probability = 0.66 * np.exp(-0.22 * ranks) + 0.04
    draw = rng.random(10)
    strong = rng.random(10) < 0.28
    return np.where(draw < hit_probability, np.where(strong, 2, 1), 0).astype(int)


def make_candidate(
    baseline: np.ndarray,
    rng: np.random.Generator,
    regression_slice: bool,
) -> np.ndarray:
    candidate = baseline.copy()
    if regression_slice and rng.random() < 0.84:
        best_position = int(np.argmax(candidate[:5]))
        destination = min(9, best_position + 5)
        candidate[best_position], candidate[destination] = candidate[destination], candidate[best_position]
        return candidate

    if rng.random() < 0.58:
        best_position = int(np.argmax(candidate))
        if best_position > 0:
            candidate[0], candidate[best_position] = candidate[best_position], candidate[0]
        elif candidate[0] == 0:
            candidate[0] = 1
    return candidate


def paired_interval(delta: np.ndarray) -> tuple[float, float]:
    spread = float(np.std(delta, ddof=1) / math.sqrt(len(delta)))
    center = float(np.mean(delta))
    return center - 1.96 * spread, center + 1.96 * spread


def generate(seed: int = 20260909, queries: int = 12000) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    metric_rows = []
    rank_rows = []

    for query_id in range(1, queries + 1):
        platform = str(rng.choice(["android", "ios", "web"], p=[0.46, 0.34, 0.20]))
        intent = str(rng.choice(["head", "torso", "tail"], p=[0.34, 0.42, 0.24]))
        translit_rate = {"head": 0.04, "torso": 0.10, "tail": 0.29}[intent]
        translit = bool(rng.random() < translit_rate)
        regression_slice = platform == "android" and intent == "tail" and translit

        baseline = make_baseline(rng)
        candidate = make_candidate(baseline, rng, regression_slice)

        row = {
            "query_id": query_id,
            "platform": platform,
            "intent": intent,
            "translit": translit,
            "regression_slice": regression_slice,
        }
        for ranker, values in (("baseline", baseline), ("candidate", candidate)):
            row[f"{ranker}_ndcg10"] = ndcg_at_k(values)
            row[f"{ranker}_mrr10"] = mrr_at_k(values)
            row[f"{ranker}_success3"] = success_at_k(values)
            row[f"{ranker}_zero_relevant10"] = float(np.sum(values) == 0)
            for rank, relevance in enumerate(values, start=1):
                rank_rows.append(
                    {
                        "query_id": query_id,
                        "platform": platform,
                        "intent": intent,
                        "translit": translit,
                        "ranker": ranker,
                        "rank": rank,
                        "relevance": int(relevance),
                    }
                )
        metric_rows.append(row)

    return pd.DataFrame(metric_rows), pd.DataFrame(rank_rows)


def summarize(metrics: pd.DataFrame) -> dict:
    ndcg_delta = metrics["candidate_ndcg10"] - metrics["baseline_ndcg10"]
    low, high = paired_interval(ndcg_delta.to_numpy())

    slices = []
    for keys, frame in metrics.groupby(["platform", "intent", "translit"], observed=True):
        delta = frame["candidate_ndcg10"] - frame["baseline_ndcg10"]
        slices.append(
            {
                "platform": keys[0],
                "intent": keys[1],
                "translit": bool(keys[2]),
                "queries": int(len(frame)),
                "baseline_ndcg10": round(float(frame["baseline_ndcg10"].mean()), 6),
                "candidate_ndcg10": round(float(frame["candidate_ndcg10"].mean()), 6),
                "delta_ndcg10": round(float(delta.mean()), 6),
            }
        )
    slices.sort(key=lambda item: item["delta_ndcg10"])

    regression = metrics[metrics["regression_slice"]]
    regression_delta = regression["candidate_ndcg10"] - regression["baseline_ndcg10"]
    overall = {
        "queries": int(len(metrics)),
        "baseline_ndcg10": round(float(metrics["baseline_ndcg10"].mean()), 6),
        "candidate_ndcg10": round(float(metrics["candidate_ndcg10"].mean()), 6),
        "delta_ndcg10": round(float(ndcg_delta.mean()), 6),
        "delta_ndcg10_ci95": [round(low, 6), round(high, 6)],
        "baseline_mrr10": round(float(metrics["baseline_mrr10"].mean()), 6),
        "candidate_mrr10": round(float(metrics["candidate_mrr10"].mean()), 6),
        "baseline_success3": round(float(metrics["baseline_success3"].mean()), 6),
        "candidate_success3": round(float(metrics["candidate_success3"].mean()), 6),
        "regression_slice_queries": int(len(regression)),
        "regression_slice_delta_ndcg10": round(float(regression_delta.mean()), 6),
    }
    decision = "limited_rollout" if overall["delta_ndcg10"] > 0 and overall["regression_slice_delta_ndcg10"] < 0 else "review"
    return {"overall": overall, "worst_slices": slices[:8], "decision": decision}


def render_report(summary: dict) -> str:
    overall = summary["overall"]
    low, high = overall["delta_ndcg10_ci95"]
    worst = summary["worst_slices"][0]
    query_count = f"{overall['queries']:,}".replace(",", " ")
    return f"""# Сравнение качества поискового ранжирования

Все данные синтетические и воспроизводимо сгенерированы. Кейс не использует данные Яндекса или другой реальной компании.

## Решение

Рекомендация — ограниченный rollout после исправления проблемного среза. На {query_count} запросах candidate-ranker повысил средний NDCG@10 с {overall['baseline_ndcg10']:.3f} до {overall['candidate_ndcg10']:.3f}: изменение {overall['delta_ndcg10']:+.3f}, 95% ДИ [{low:+.3f}; {high:+.3f}]. MRR@10 изменился с {overall['baseline_mrr10']:.3f} до {overall['candidate_mrr10']:.3f}.

При этом в срезе Android × tail × translit ({overall['regression_slice_queries']} запросов) NDCG@10 изменился на {overall['regression_slice_delta_ndcg10']:+.3f}. Худший наблюдаемый срез: {worst['platform']} × {worst['intent']} × translit={worst['translit']}, delta {worst['delta_ndcg10']:+.3f}.

## Метод

- Единица анализа — запрос.
- Primary metric — NDCG@10; дополнительные метрики — MRR@10, Success@3 и zero-relevant@10.
- Сравнение парное: для каждого запроса доступны позиции baseline и candidate.
- Отдельно проверяются platform, intent и translit, чтобы агрегат не скрывал локальную регрессию.
- SQL-версия расчёта хранится в `sql/ranking_metrics.sql`, Python-версия — в `src/pipeline.py`.

## Следующие шаги

1. Исправить обработку транслитерации для длинного хвоста Android-запросов.
2. Повторить offline evaluation на замороженном наборе relevance judgments.
3. После прохождения slice-guardrail провести online A/B-тест с CTR, reformulation rate и long-click guardrails.
"""


def main() -> None:
    data_dir = ROOT / "data/generated"
    output_dir = ROOT / "outputs"
    report_dir = ROOT / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics, ranking_rows = generate()
    metrics.to_csv(data_dir / "query_metrics.csv", index=False)
    ranking_rows.to_csv(data_dir / "ranking_rows.csv", index=False)
    summary = summarize(metrics)
    (output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (report_dir / "analysis_report.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
