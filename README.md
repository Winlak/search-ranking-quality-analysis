# Сравнение качества поискового ранжирования

Портфолио-кейс продуктового аналитика о выборе между baseline и candidate-ranker. Все данные синтетические и воспроизводимо генерируются из кода; это не данные Яндекса или другой реальной компании.

## Бизнес-вопрос

Можно ли запускать новую модель ранжирования, если общая offline-метрика растёт, но часть запросов может деградировать?

Кейс рассчитывает NDCG@10, MRR@10, Success@3 и zero-relevant@10, использует парное сравнение на уровне запроса и отдельно проверяет platform × intent × translit. Итоговый вывод строится из фактического результата pipeline в [аналитическом отчёте](reports/analysis_report.md).

## Структура

```text
src/pipeline.py          генерация, метрики, срезы и отчёт
sql/ranking_metrics.sql SQL-расчёт метрик на уровне запроса
tests/                   unit-тесты формул
outputs/metrics.json     итоговые числа и худшие срезы
reports/                 decision-ready вывод
data/README.md           гранулярность и происхождение данных
```

## Запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python -m src.pipeline
```

## Для резюме

Сравнил baseline и candidate-ranker на синтетическом наборе поисковых запросов: реализовал NDCG@10, MRR@10 и Success@3 в Python и SQL, провёл парный анализ и slice-check по platform × intent × translit, обнаружил скрытую агрегатом Android-регрессию и сформулировал безопасный план rollout.

