"""Table B: how concentrated the purchases of a label week are (results/t_diag_concentration.csv).

One row per fold. A "purchase" is one label pair: a de-duplicated
(customer, article) of the fold's label file (hmrec.eval.diagnostic_labels.label_pair_counts).

    top{N}_pair_share        share of the week's label pairs on the N articles with
                             the most label pairs that week
    first_sale_within_{d}d   articles whose FIRST sale is in [anchor - d, anchor):
                             how many, and their share of the label pairs
    first_sale_in_label_week articles with NO sale before the anchor: how many,
                             and their share of the label pairs

"First sale" comes from the purchase lines with t_dat < anchor only
(hmrec.data.asof.history_window). An article of the label file without any such
line was first sold on the anchor day or later; nothing after the anchor is read
to know that. These numbers describe the label week itself. They are NOT
features and enter no model.
"""

from dataclasses import dataclass

import polars as pl

from hmrec.data.asof import TRANSACTION_DAY
from hmrec.data.id_maps import ARTICLE_IDX
from hmrec.diagnostics.records import float_text
from hmrec.eval.diagnostic_labels import N_LABEL_PAIRS

FIRST_SALE_DAY = "first_sale_day"


@dataclass(frozen=True)
class Concentration:
    """The concentration numbers of one label week.

    Attributes:
        n_label_pairs: Label pairs of the week.
        n_label_articles: Distinct articles with a label pair.
        top_pair_shares: N -> share of the label pairs on the N most bought articles.
        n_new_articles: Label articles first sold in [anchor - d, anchor).
        new_pair_share: Their share of the label pairs.
        n_unsold_before_articles: Label articles with no sale before the anchor.
        unsold_before_pair_share: Their share of the label pairs.
    """

    n_label_pairs: int
    n_label_articles: int
    top_pair_shares: dict[int, float]
    n_new_articles: int
    new_pair_share: float
    n_unsold_before_articles: int
    unsold_before_pair_share: float


def first_sale_days(lines: pl.DataFrame) -> pl.DataFrame:
    """The first sale day of every article that has a purchase line.

    Args:
        lines: One row per purchase line BEFORE the anchor (from
            hmrec.data.asof.history_window). Columns: article_idx (int32),
            t_dat_day (int16).

    Returns:
        One row per article in `lines`. Columns: article_idx (int32),
        first_sale_day (int16, the smallest t_dat_day); sorted by article_idx.
    """
    first = lines.group_by(ARTICLE_IDX).agg(pl.col(TRANSACTION_DAY).min().alias(FIRST_SALE_DAY))
    return first.sort(ARTICLE_IDX)


def top_pair_share(label_counts: pl.DataFrame, top_n: int) -> float:
    """Share of the label pairs on the `top_n` articles with the most label pairs.

    Ties are broken by article_idx ascending, so the result never depends on
    the row order. With fewer than top_n articles the share is 1.

    Args:
        label_counts: One row per article: article_idx, n_label_pairs (> 0).
        top_n: N, > 0.
    """
    if top_n <= 0:
        raise ValueError(f"top_n must be > 0, got {top_n}")
    ordered = label_counts.sort([N_LABEL_PAIRS, ARTICLE_IDX], descending=[True, False])
    top_pairs = int(ordered.head(top_n)[N_LABEL_PAIRS].sum())
    return top_pairs / int(label_counts[N_LABEL_PAIRS].sum())


def _check_before_anchor(first_sale: pl.DataFrame, anchor_day: int) -> None:
    """Fail if a first sale day is not before the anchor (it would be label-week data)."""
    latest = first_sale[FIRST_SALE_DAY].max()
    if isinstance(latest, int) and latest >= anchor_day:
        raise ValueError(
            f"first_sale_day contains {latest}, which is not before the anchor day {anchor_day}"
        )


def concentration(
    label_counts: pl.DataFrame,
    first_sale: pl.DataFrame,
    anchor_day: int,
    top_article_counts: tuple[int, ...],
    new_article_days: int,
) -> Concentration:
    """The concentration numbers of one label week.

    Args:
        label_counts: One row per article with a label: article_idx, n_label_pairs.
        first_sale: One row per article sold BEFORE the anchor: article_idx,
            first_sale_day (< anchor_day). Articles never sold before are absent.
        anchor_day: Day index of the fold's anchor.
        top_article_counts: The N of the top-N shares.
        new_article_days: d of the window [anchor_day - d, anchor_day).

    Raises:
        ValueError: If there is no label, or a first sale day is not before the anchor.
    """
    if label_counts.height == 0:
        raise ValueError("the label week has no label pairs")
    _check_before_anchor(first_sale, anchor_day)
    total = int(label_counts[N_LABEL_PAIRS].sum())
    # A left join keeps every label article; first_sale_day is null for an article
    # with no purchase line before the anchor (first sold in the label week or later).
    joined = label_counts.join(first_sale, on=ARTICLE_IDX, how="left")
    unsold_before = joined.filter(pl.col(FIRST_SALE_DAY).is_null())
    # Window [anchor_day - d, anchor_day): the anchor day itself is the label week.
    new = joined.filter(pl.col(FIRST_SALE_DAY) >= anchor_day - new_article_days)
    return Concentration(
        n_label_pairs=total,
        n_label_articles=label_counts.height,
        top_pair_shares={n: top_pair_share(label_counts, n) for n in top_article_counts},
        n_new_articles=new.height,
        new_pair_share=int(new[N_LABEL_PAIRS].sum()) / total,
        n_unsold_before_articles=unsold_before.height,
        unsold_before_pair_share=int(unsold_before[N_LABEL_PAIRS].sum()) / total,
    )


def concentration_columns(
    top_article_counts: tuple[int, ...], new_article_days: int
) -> tuple[str, ...]:
    """The columns of table B (without the provenance columns)."""
    window = f"first_sale_within_{new_article_days}d"
    return (
        "fold",
        "phase",
        "anchor",
        "n_label_pairs",
        "n_label_articles",
        *(f"top{n}_pair_share" for n in top_article_counts),
        f"n_articles_{window}",
        f"pair_share_{window}",
        "n_articles_first_sale_in_label_week",
        "pair_share_first_sale_in_label_week",
    )


def concentration_row(
    fold: str, phase: str, anchor: str, stats: Concentration, new_article_days: int
) -> dict[str, str]:
    """One row of table B (the anchor as YYYY-MM-DD text)."""
    window = f"first_sale_within_{new_article_days}d"
    return {
        "fold": fold,
        "phase": phase,
        "anchor": anchor,
        "n_label_pairs": str(stats.n_label_pairs),
        "n_label_articles": str(stats.n_label_articles),
        **{f"top{n}_pair_share": float_text(s) for n, s in stats.top_pair_shares.items()},
        f"n_articles_{window}": str(stats.n_new_articles),
        f"pair_share_{window}": float_text(stats.new_pair_share),
        "n_articles_first_sale_in_label_week": str(stats.n_unsold_before_articles),
        "pair_share_first_sale_in_label_week": float_text(stats.unsold_before_pair_share),
    }
