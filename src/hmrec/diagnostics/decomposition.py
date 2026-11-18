"""Table A: where each fold's MAP@12 comes from (results/t_diag_decomposition.csv).

One row per fold, from the recorded result tables only (no label is read):

    union_recall                       share of the labels the six channels found
    {system}_recall@12                 share of the labels in the system's Top-12
    {system}_recall@12_over_union      recall@12 / union_recall: of the labels the
                                       candidates contain, the share ranked into the Top-12
    {system}_map@12                    the system's MAP@12

for the two systems "lambdamart" (the locked Ranker) and "rrf_baseline". Both
rank the SAME candidates, so union_recall is the same for both.
"""

from hmrec.diagnostics.recorded import RecordedFold
from hmrec.diagnostics.records import float_text

LAMBDAMART = "lambdamart"
RRF_BASELINE_SYSTEM = "rrf_baseline"
SYSTEMS = (LAMBDAMART, RRF_BASELINE_SYSTEM)


def ratio_column(system: str, recall_metric: str) -> str:
    """Name of a system's "recall@12 / union recall" column."""
    return f"{system}_{recall_metric}_over_union"


def decomposition_columns(map_metric: str, recall_metric: str) -> tuple[str, ...]:
    """The columns of table A (without the provenance columns)."""
    per_system = [
        column
        for system in SYSTEMS
        for column in (
            f"{system}_{recall_metric}",
            ratio_column(system, recall_metric),
            f"{system}_{map_metric}",
        )
    ]
    return ("fold", "phase", "n_queries", "union_recall", *per_system)


def recall_over_union(recall: float, union_recall: float) -> float:
    """recall@K / union recall: the share of the found labels that reached the Top-K.

    Both numbers are means over the same queries, so this is a ratio of means.

    Raises:
        ValueError: If union_recall is not positive (no label was found at all).
    """
    if union_recall <= 0:
        raise ValueError(f"union recall must be > 0 to take a ratio, got {union_recall}")
    return recall / union_recall


def decomposition_row(
    recorded: RecordedFold, map_metric: str, recall_metric: str
) -> dict[str, str]:
    """One row of table A from a fold's recorded numbers."""
    row = {
        "fold": recorded.fold,
        "phase": recorded.phase,
        "n_queries": str(recorded.n_queries),
        "union_recall": float_text(recorded.union_recall),
    }
    for system, scores in ((LAMBDAMART, recorded.ranker), (RRF_BASELINE_SYSTEM, recorded.baseline)):
        recall = scores[recall_metric]
        row[f"{system}_{recall_metric}"] = float_text(recall)
        row[ratio_column(system, recall_metric)] = float_text(
            recall_over_union(recall, recorded.union_recall)
        )
        row[f"{system}_{map_metric}"] = float_text(scores[map_metric])
    return row
