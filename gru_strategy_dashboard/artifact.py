"""Build the compact interactive dashboard from measured strategy results."""

from __future__ import annotations

from typing import Any


STRATEGY_SOURCE_ID = "teacher_strategy_rollouts"
CHECKPOINT_SOURCE_ID = "teacher_checkpoint"
MAP_POINTS_PER_STYLE_AND_TABLE = 64


def _compact_memory_map(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep an even, responsive dot sample below the portable row limit."""

    compacted: list[dict[str, Any]] = []
    counts: dict[tuple[str, int], int] = {}
    for index in range(0, len(rows), 2):
        pair = rows[index : index + 2]
        if len(pair) != 2:
            continue
        specific = pair[0]
        key = (str(specific["action_style"]), int(specific["players"]))
        count = counts.get(key, 0)
        if count >= MAP_POINTS_PER_STYLE_AND_TABLE:
            continue
        counts[key] = count + 1
        compacted.extend(pair)
    return compacted


def _source_specs(summary: dict[str, Any], summary_path: str) -> list[dict[str, Any]]:
    checkpoint = summary["checkpoint"]
    protocol = summary["protocol"]
    return [
        {
            "id": STRATEGY_SOURCE_ID,
            "label": "Measured teacher choices",
            "path": summary_path,
            "query": {
                "engine": "DuckDB over the measured JSON snapshot",
                "sql": f"SELECT *\nFROM read_json_auto({summary_path!r})",
                "analysis_command": (
                    "PYTHONPATH=. python -m gru_strategy_dashboard.generate "
                    f"--games-per-player-count {protocol['games_per_player_count_per_style']} "
                    f"--seed {protocol['base_seed']} --device {protocol['device']} "
                    f"--max-turns {protocol['max_turns']} "
                    f"--torch-threads {protocol['torch_threads']}"
                ),
                "tables_used": [summary_path, checkpoint["path"]],
                "description": (
                    "The checked-in runner loaded the named teacher, played paired "
                    "self-play deals, and wrote the compact rows shown here."
                ),
                "executed_at": summary["generated_at"],
                "filters": [
                    f"Player counts: {', '.join(map(str, protocol['player_counts']))}.",
                    (
                        f"{protocol['games_per_player_count_per_style']} games per "
                        "table size and move style."
                    ),
                    "All four ordinary suits are combined by card number.",
                    "Every seat uses the same teacher.",
                ],
                "metric_definitions": [
                    protocol["hold_definition"],
                    protocol["memory_layer_test"],
                    protocol["input_pull_test"],
                    protocol["memory_map"],
                ],
            },
        },
        {
            "id": CHECKPOINT_SOURCE_ID,
            "label": (
                f"Puffer GRU teacher, epoch {checkpoint['epoch']} - SHA-256 "
                f"{checkpoint['sha256'][:12]}"
            ),
            "path": checkpoint["path"],
        },
        {
            "id": "analysis_code",
            "label": "Reproducible strategy checks",
            "path": "gru_strategy_dashboard/analyze.py",
        },
        {
            "id": "safe_input_contract",
            "label": "Input definitions",
            "path": "temporal_observation.py",
        },
        {
            "id": "model_code",
            "label": "Teacher model and loader",
            "path": "temporal_puffer_v4.py",
        },
    ]


def _filters() -> list[dict[str, Any]]:
    datasets = (
        "rank_choices",
        "effect_cards_by_rival_distance",
        "memory_layer_tests",
        "input_pull",
        "memory_map",
        "model_shape",
    )
    return [
        {
            "id": "action_style",
            "label": "Move style",
            "dataset": "headline",
            "field": "action_style",
            "defaultValue": "Varied choices",
            "includeAll": False,
            "targets": [
                {"dataset": dataset, "field": "action_style"}
                for dataset in datasets
            ],
        },
        {
            "id": "view_players",
            "label": "Table size",
            "dataset": "headline",
            "field": "view_players",
            "defaultValue": "All table sizes",
            "includeAll": False,
            "targets": [
                {"dataset": dataset, "field": "view_players"}
                for dataset in datasets
            ],
        },
    ]


def _cards() -> list[dict[str, Any]]:
    return [
        {
            "id": "games",
            "description": "Completed self-play games in the selected view.",
            "dataset": "headline",
            "sourceId": STRATEGY_SOURCE_ID,
            "metrics": [{"label": "Games", "field": "games", "format": "compact"}],
        },
        {
            "id": "card_decisions",
            "description": "Turns where the teacher chose a card or passed.",
            "dataset": "headline",
            "sourceId": STRATEGY_SOURCE_ID,
            "metrics": [
                {"label": "Card decisions", "field": "card_decisions", "format": "compact"}
            ],
        },
        {
            "id": "strategic_pass",
            "description": "Passes made while at least one card could be played.",
            "dataset": "headline",
            "sourceId": STRATEGY_SOURCE_ID,
            "metrics": [
                {
                    "label": "Passed with a playable card",
                    "field": "strategic_pass_rate",
                    "format": "percent",
                }
            ],
        },
        {
            "id": "memory_change",
            "description": "Choices where clearing all stored history changed the favorite move.",
            "dataset": "headline",
            "sourceId": STRATEGY_SOURCE_ID,
            "metrics": [
                {
                    "label": "History changed the favorite",
                    "field": "memory_change_rate",
                    "format": "percent",
                }
            ],
        },
    ]


def _chart(
    *,
    chart_id: str,
    title: str,
    subtitle: str,
    chart_type: str,
    dataset: str,
    x: dict[str, Any],
    y: dict[str, Any],
    tooltip: list[dict[str, Any]],
    palette: dict[str, str],
    color: dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
    legend: dict[str, Any] | None = None,
    layout: str = "full",
) -> dict[str, Any]:
    encodings: dict[str, Any] = {"x": x, "y": y, "tooltip": tooltip}
    if color is not None:
        encodings["color"] = color
    result: dict[str, Any] = {
        "id": chart_id,
        "title": title,
        "subtitle": subtitle,
        "type": chart_type,
        "intent": "relationship" if chart_type == "scatter" else "comparison",
        "question": title,
        "rationale": "The view keeps the main pattern visible and exact counts in hover details.",
        "dataset": dataset,
        "sourceId": STRATEGY_SOURCE_ID,
        "encodings": encodings,
        "layout": layout,
        "palette": palette,
    }
    if settings:
        result["settings"] = settings
    if legend:
        result["legend"] = legend
    return result


def _charts() -> list[dict[str, Any]]:
    return [
        _chart(
            chart_id="rank_hold",
            title="How often each card number is saved",
            subtitle=(
                "Saved means that number could be played, but the teacher chose "
                "something else. All suits are combined."
            ),
            chart_type="horizontalBar",
            dataset="rank_choices",
            x={"field": "rank_with_effect", "type": "nominal", "label": "Card number"},
            y={"field": "hold_rate", "type": "quantitative", "label": "Saved", "format": "percent"},
            tooltip=[
                {"field": "opportunities", "type": "quantitative", "label": "Times it could be played"},
                {"field": "play_rate_with_1_2_cards", "type": "quantitative", "label": "Played with 1-2 cards left", "format": "percent"},
                {"field": "rival_pressure_change_pp", "type": "quantitative", "label": "Use change when a rival is close", "unit": "points"},
            ],
            palette={"kind": "sequential", "name": "blue"},
            settings={"orientation": "horizontal", "sort": "descending"},
        ),
        _chart(
            chart_id="effect_cards_pressure",
            title="Effect cards when a rival is close",
            subtitle="How often each effect card is played when legal, split by the closest rival's hand.",
            chart_type="bar",
            dataset="effect_cards_by_rival_distance",
            x={"field": "rank", "type": "nominal", "label": "Effect card"},
            y={"field": "play_rate", "type": "quantitative", "label": "Played", "format": "percent"},
            color={"field": "rival_distance", "type": "nominal", "label": "Closest rival"},
            tooltip=[
                {"field": "opportunities", "type": "quantitative", "label": "Times it could be played"}
            ],
            palette={"kind": "categorical", "name": "blue-orange"},
            settings={"groupMode": "grouped"},
            legend={"position": "bottom", "title": "Closest rival"},
        ),
        _chart(
            chart_id="memory_stages",
            title="Which part of stored history changes the favorite?",
            subtitle="The hand, legal moves, and newest public updates stay exactly the same.",
            chart_type="bar",
            dataset="memory_layer_tests",
            x={"field": "memory_test", "type": "ordinal", "label": "What is cleared"},
            y={"field": "favorite_change_rate", "type": "quantitative", "label": "Favorite changed", "format": "percent"},
            tooltip=[
                {"field": "decisions", "type": "quantitative", "label": "Matched decisions"},
                {"field": "move_weight_shift", "type": "quantitative", "label": "How much the choice mix moved", "format": "percent"},
            ],
            palette={"kind": "sequential", "name": "blue"},
        ),
        _chart(
            chart_id="input_sway",
            title="Which part of the current turn can sway a close choice?",
            subtitle=(
                "Measured near sampled decisions and adjusted for how much each "
                "input usually changes."
            ),
            chart_type="horizontalBar",
            dataset="input_pull",
            x={"field": "input_part", "type": "nominal", "label": "Part of the current turn"},
            y={"field": "share_of_local_pull", "type": "quantitative", "label": "Share of measured sway", "format": "percent"},
            tooltip=[
                {"field": "sampled_decisions", "type": "quantitative", "label": "Sampled decisions"},
                {"field": "values", "type": "quantitative", "label": "Numbers in this part"},
            ],
            palette={"kind": "sequential", "name": "orange"},
            settings={"orientation": "horizontal", "sort": "descending"},
        ),
        _chart(
            chart_id="memory_map",
            title="Map of similar remembered situations",
            subtitle=(
                "Nearby dots have more similar stored values. Hover to see the hand, "
                "closest rival, and favorite move."
            ),
            chart_type="scatter",
            dataset="memory_map",
            x={"field": "memory_direction_1", "type": "quantitative", "label": "Map direction 1"},
            y={"field": "memory_direction_2", "type": "quantitative", "label": "Map direction 2"},
            color={"field": "rival_pressure", "type": "nominal", "label": "Closest rival"},
            tooltip=[
                {"field": "players", "type": "quantitative", "label": "Players"},
                {"field": "cards_in_hand", "type": "quantitative", "label": "Cards it holds"},
                {"field": "closest_rival_cards", "type": "quantitative", "label": "Closest rival's cards"},
                {"field": "favorite_move", "type": "nominal", "label": "Favorite move"},
                {"field": "move_taken", "type": "nominal", "label": "Move taken"},
                {"field": "decision_number", "type": "quantitative", "label": "Decision number"},
            ],
            palette={"kind": "categorical", "name": "blue-orange"},
            legend={"position": "bottom", "title": "Closest rival"},
        ),
    ]


def _tables() -> list[dict[str, Any]]:
    return [
        {
            "id": "model_shape",
            "title": "Model shape: what goes in, what is stored, what comes out",
            "subtitle": "The first size changes only when more players are checked together.",
            "dataset": "model_shape",
            "sourceId": STRATEGY_SOURCE_ID,
            "defaultSort": {"field": "stage", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "stage", "label": "Part", "type": "text"},
                {"field": "exact_shape", "label": "Exact shape", "type": "text"},
                {"field": "plain_meaning", "label": "Plain meaning", "type": "text"},
            ],
        }
    ]


def _blocks(summary: dict[str, Any]) -> list[dict[str, Any]]:
    findings = "\n".join(f"- {finding}" for finding in summary["findings"])
    protocol = summary["protocol"]
    memory_directions = summary["datasets"]["memory_map_explained"][0]
    total_games = (
        len(protocol["player_counts"])
        * protocol["games_per_player_count_per_style"]
        * len(protocol["action_styles"])
    )
    return [
        {
            "id": "intro",
            "type": "markdown",
            "body": (
                "## The teacher at a glance\n\n"
                "Pick a move style and table size above. Suits are always combined, "
                "so the page compares card numbers only."
            ),
        },
        {
            "id": "headline",
            "type": "metric-strip",
            "cardIds": ["games", "card_decisions", "strategic_pass", "memory_change"],
        },
        {
            "id": "findings",
            "type": "markdown",
            "sourceId": STRATEGY_SOURCE_ID,
            "body": "## Four things worth noticing\n\n" + findings,
        },
        {"id": "rank_hold_block", "type": "chart", "chartId": "rank_hold"},
        {"id": "effect_pressure_block", "type": "chart", "chartId": "effect_cards_pressure"},
        {
            "id": "inside_note",
            "type": "markdown",
            "body": (
                "## What changes the choice?\n\n"
                "These checks alter one thing at a time on the same decision. They show "
                "what can change the result, not a hidden label inside the model."
            ),
        },
        {"id": "memory_stages_block", "type": "chart", "chartId": "memory_stages"},
        {"id": "input_sway_block", "type": "chart", "chartId": "input_sway"},
        {
            "id": "memory_map_note",
            "type": "markdown",
            "sourceId": STRATEGY_SOURCE_ID,
            "body": (
                "## Browse remembered situations\n\n"
                f"The two map directions keep {memory_directions['first_direction']:.1%} "
                f"and {memory_directions['second_direction']:.1%} of the visible variation. "
                "They are display directions, not named ideas."
            ),
        },
        {"id": "memory_map_block", "type": "chart", "chartId": "memory_map"},
        {"id": "shape_block", "type": "table", "tableId": "model_shape", "layout": "full"},
        {
            "id": "method",
            "type": "markdown",
            "sourceId": STRATEGY_SOURCE_ID,
            "body": (
                "## What this page is based on\n\n"
                f"{total_games:,} self-play games and "
                f"{protocol['interpretability_samples']:,} sampled memories. Every seat "
                "used the same teacher. The findings describe these games; they do not "
                "prove that one event caused another."
            ),
        },
    ]


def build_artifact(
    summary: dict[str, Any],
    *,
    summary_path: str = "benchmarks/gru_strategy/strategy_summary.json",
) -> dict[str, Any]:
    """Return the compact dashboard payload used by the portable builder."""

    sources = _source_specs(summary, summary_path)
    visible_datasets = {
        name: summary["datasets"][name]
        for name in (
            "headline",
            "rank_choices",
            "effect_cards_by_rival_distance",
            "memory_layer_tests",
            "input_pull",
            "memory_map",
            "model_shape",
        )
    }
    visible_datasets["memory_map"] = _compact_memory_map(
        summary["datasets"]["memory_map"]
    )
    visible_datasets["model_shape"] = [
        {
            **shape,
            "action_style": headline["action_style"],
            "view_players": headline["view_players"],
        }
        for headline in summary["datasets"]["headline"]
        for shape in summary["datasets"]["model_shape"]
    ]
    return {
        "surface": "dashboard",
        "manifest": {
            "version": 1,
            "surface": "dashboard",
            "title": "Inside the Puffer GRU teacher",
            "description": (
                "A compact, interactive look at card saving, rival pressure, stored "
                "history, current-turn influence, and exact model shapes."
            ),
            "generatedAt": summary["generated_at"],
            "filters": _filters(),
            "cards": _cards(),
            "charts": _charts(),
            "tables": _tables(),
            "sources": sources,
            "blocks": _blocks(summary),
        },
        "snapshot": {
            "version": 1,
            "generatedAt": summary["generated_at"],
            "status": "ready",
            "datasets": visible_datasets,
        },
        "sources": sources,
    }
