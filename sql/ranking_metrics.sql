WITH ranked AS (
    SELECT
        query_id,
        platform,
        intent,
        translit,
        ranker,
        rank,
        relevance,
        (POWER(2, relevance) - 1) / LOG2(rank + 1) AS dcg_part
    FROM ranking_rows
),
query_dcg AS (
    SELECT
        query_id,
        platform,
        intent,
        translit,
        ranker,
        SUM(dcg_part) AS dcg10,
        MIN(CASE WHEN relevance > 0 THEN rank END) AS first_relevant_rank,
        MAX(CASE WHEN rank <= 3 AND relevance > 0 THEN 1 ELSE 0 END) AS success3
    FROM ranked
    GROUP BY 1, 2, 3, 4, 5
),
ideal AS (
    SELECT
        query_id,
        SUM((POWER(2, relevance) - 1) / LOG2(ideal_rank + 1)) AS idcg10
    FROM (
        SELECT
            query_id,
            relevance,
            ROW_NUMBER() OVER (PARTITION BY query_id ORDER BY relevance DESC, doc_id) AS ideal_rank
        FROM relevance_judgments
    ) ordered_relevance
    GROUP BY 1
)
SELECT
    q.query_id,
    q.platform,
    q.intent,
    q.translit,
    q.ranker,
    CASE WHEN i.idcg10 = 0 THEN 0 ELSE q.dcg10 / i.idcg10 END AS ndcg10,
    CASE WHEN q.first_relevant_rank IS NULL THEN 0 ELSE 1.0 / q.first_relevant_rank END AS mrr10,
    q.success3
FROM query_dcg q
JOIN ideal i USING (query_id);
