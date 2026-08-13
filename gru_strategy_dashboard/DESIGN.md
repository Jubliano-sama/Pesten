# GRU strategy dashboard design contract

## Brief

- Audience: a Pesten player who wants to understand the teacher without ML
  vocabulary.
- Question: which card numbers are saved, when are they used, what public
  context changes choices, which stored-history layers matter, what clusters
  inside memory, and what exactly enters and leaves the model?
- Surface: one self-contained interactive HTML dashboard.
- Filters: move-picking style and table size.
- Source: fresh paired self-play from the current teacher checkpoint.

## Metric roles

- Headline: games, card decisions, deliberate passes, and choices changed by
  stored history.
- Main views: card-number saving, effect-card use under pressure, matched
  stored-history clearing, current-turn sway, and a map of similar memories.
- Detail: exact counts and ranges stay in hover details; exact model shapes use
  one five-row table.
- Guardrail: all suits pooled; every seat uses the teacher; results are
  descriptive and never presented as causal.

## Chart map

| Section | Question | Family / native type | Main fields | Palette |
| --- | --- | --- | --- | --- |
| Holding | Which numbers are saved? | Comparison / `horizontalBar` | rank, hold rate, chances | One blue root |
| Rival pressure | Do effect cards change near a rival win? | Comparison / grouped `bar` | rank, play rate, rival band | Blue and orange roots |
| Stored history | Which stage changes the result? | Comparison / `bar` | stage cleared, favorite change | One blue root |
| Current turn | Which part can sway a close choice? | Comparison / `horizontalBar` | input part, share of sway | One orange root |
| Memory map | Which internal situations look alike? | Relationship / `scatter` | two display directions, rival band | Blue and orange roots |
| Model shape | What goes in, is stored, and comes out? | Five-row table | part, exact shape, meaning | None |

## Language rules

Visible labels use “card number,” “cards it holds,” “stored history,” “favorite
move,” and “what can sway the choice.” Exact shape notation appears only in the
small table the user requested.
