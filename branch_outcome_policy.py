"""Public action scoring for the counterfactual rollout learner."""

from pathlib import Path

import torch
from torch import nn

from strongest_policy_search import NativeSuitOrbitCore
from train_outcome_residual import PublicHistoryCore, SPECIAL, file_hash, public_features, teacher


def legal_mask(obs, single_card=None):
    """Reconstruct the normal rule mask and optionally apply the adapter's exact retry mask."""
    batch = len(obs)
    cards = torch.arange(50, device=obs.device)
    suit = torch.where(cards < 48, cards // 12, 4)
    rank = torch.where(cards < 48, 1 + cards % 12, torch.where(cards == 48, 13, 0))
    current_suit = obs[:, 50:55].argmax(1)[:, None]
    current_rank = obs[:, 55:69].argmax(1)[:, None]
    matches = (suit == current_suit) | (rank == current_rank) | (cards >= 48)
    valid_card = (obs[:, :50] > 0) & torch.where(obs[:, 90:91] > 0, rank == current_rank, matches)
    if single_card is not None:
        restricted = single_card[:, None] != 0
        valid_card &= ~restricted | (cards[None] == single_card[:, None] - 1)
    suit_phase = obs[:, 93:94] > 0
    return torch.cat((valid_card & ~suit_phase, suit_phase.expand(batch, 4), ~suit_phase), 1)


def _decision_features(decision):
    # Only a draw retry can narrow the ordinary card mask. Carry the legal ID
    # on the decision token and strip it before the recurrent teacher sees it.
    playable = [a for a, legal in enumerate(decision.legal_actions[:50]) if legal]
    marker = -1 if not playable else (playable[0] + 1 if len(playable) == 1 else 0)
    return (*decision.observation.values, 0.0, 0.0, 0.0, 0.0, float(marker), 1.0)


class BranchActionHead(nn.Module):
    """An action head equivariant to suits and the eight interchangeable ordinary ranks."""

    def __init__(self, hidden=24, graph=True, style_heads=False, frequencies=None):
        super().__init__()
        self.graph = graph
        self.style_heads = style_heads
        self.register_buffer("frequencies", frequencies)
        self.register_buffer("table_offsets", None)
        self.state = nn.Linear(38, 48)
        self.action = nn.Linear(30, 48, bias=False)
        if graph:
            self.node = nn.Sequential(
                nn.Linear(13 + (21 if frequencies is not None else 0), hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.suit = nn.Linear(hidden, hidden, bias=False)
            self.rank = nn.Linear(hidden, hidden, bias=False)
            self.context = nn.Linear(38, hidden)
            self.card = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
            self.action_node = nn.Linear(hidden, 48, bias=False)
        self.output = nn.Sequential(
            nn.Tanh(), nn.Linear(48, 32), nn.Tanh(), nn.Linear(32, 2 if style_heads else 1)
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def node_values(self, obs, seen, state, tracking):
        cards = obs[:, :48].reshape(-1, 4, 12)
        recent = seen[:, :48].reshape(-1, 4, 12)
        top_suit = obs[:, 50:54, None].expand_as(cards)
        top_rank = obs[:, None, 56:68].expand_as(cards)
        ranks = torch.arange(12, device=obs.device)
        special = torch.stack([ranks == r for r in SPECIAL], -1).to(obs.dtype)
        node = torch.cat(
            [
                torch.stack(
                    [
                        cards,
                        recent,
                        cards.sum(2, keepdim=True).expand_as(cards) / 6,
                        cards.sum(1, keepdim=True).expand_as(cards) / 3,
                        recent.sum(2, keepdim=True).expand_as(cards) / 8,
                        recent.sum(1, keepdim=True).expand_as(cards) / 3,
                        top_suit,
                        top_rank,
                        cards * (top_suit + top_rank).clamp(max=1),
                    ],
                    -1,
                ),
                special.expand(len(obs), 4, -1, -1),
            ],
            -1,
        )
        if self.frequencies is not None:
            from branch_public_belief import belief_nodes

            node = torch.cat((node, belief_nodes(obs, tracking, self.frequencies)), dim=-1)
        emb = self.node(node)
        held = cards[:, :, :, None]
        suit = self.suit((emb * held).sum(2, keepdim=True) / 3)
        rank = self.rank((emb * held).sum(1, keepdim=True) / 2)
        message = self.card(emb + suit + rank + self.context(state)[:, None, None, :])
        suit_emb = (message * held).sum(2) / 3
        global_emb = suit_emb.mean(1, keepdim=True)
        return torch.cat(
            (message.flatten(1, 2), global_emb.expand(-1, 2, -1), suit_emb, global_emb), 1
        )

    def embeddings(self, obs, seen, value, rows, act, tracking=None):
        state, action = public_features(obs, seen, value, (rows, act))
        representation = self.state(state)[rows] + self.action(action)
        if self.graph:
            nodes = self.node_values(obs, seen, state, tracking)
            representation = representation + self.action_node(nodes[rows, act])
        # The final linear map can also be calibrated on full-game paired outcomes.
        hidden = self.output[0](representation)
        hidden = self.output[1](hidden)
        return self.output[2](hidden)

    def forward(self, obs, seen, value, mask, random_probability=None, tracking=None):
        rows, act = mask.nonzero(as_tuple=True)
        features = self.embeddings(obs, seen, value, rows, act, tracking)
        out = torch.zeros_like(mask, dtype=obs.dtype)
        estimates = self.output[3](features)
        if self.table_offsets is not None:
            offsets = self.table_offsets[(obs[rows, 92].long() - 2).clamp(0, 2)]
            estimates = estimates + (offsets * features[:, None, :]).sum(-1)
        if self.style_heads:
            if random_probability is None:
                raise ValueError("two style-conditioned heads need an opponent posterior")
            estimates = torch.lerp(estimates[:, 0], estimates[:, 1], random_probability[rows])
        out[rows, act] = estimates.flatten()
        return out


class BranchOutcomeCore(PublicHistoryCore):
    def __init__(self, parent, head, scale, margin, style=None):
        super().__init__(parent)
        self.head = head
        self.scale = scale
        self.margin = margin
        self.style = style

    def initial_state(self, batch_size, device):
        regular = super().initial_state(batch_size, device)
        if self.style is not None:
            regular = (*regular, self.style.initial_state(batch_size, device))
        if self.head.frequencies is not None:
            from branch_public_belief import initial_history

            regular = (*regular, initial_history(batch_size, device))
        return regular

    def forward_events_eval(self, tokens, state):
        extra = ()
        if self.head.frequencies is not None:
            from branch_public_belief import advance_history

            extra = (advance_history(tokens, state[-1]),)
            state = state[:-1]
        if self.style is not None:
            extra = (state[-1], *extra)
            state = state[:-1]
        return (*super().forward_events_eval(tokens, state), *extra)

    def forward_eval(self, tokens, state):
        decision = tokens[:, 99] == 1
        posterior = None
        extra_state = ()
        tracking = None
        if self.head.frequencies is not None:
            from branch_public_belief import advance_history

            h = advance_history(tokens, state[-1])
            extra_state = (h,)
            tracking = h[0, decision]
            state = state[:-1]
        if self.style is not None:
            probabilities, style_state = self.style.observe(tokens, state[-1])
            posterior = torch.from_numpy(probabilities).to(tokens)[decision]
            extra_state = (style_state, *extra_state)
            state = state[:-1]
        inputs = tokens.clone()
        inputs[decision, 98] = 0
        logits, value, next_state = self.core.forward_eval(inputs, state[:-1])
        counts = self.played_cards(tokens, state[-1])
        if decision.any():
            obs = tokens[decision]
            legal = legal_mask(obs, obs[:, 98].long())
            prior = logits[decision]
            top = prior.masked_fill(~legal, -torch.inf).argmax(1, keepdim=True)
            supported = legal & (prior >= prior.gather(1, top) - self.margin)
            scores = self.head(
                obs, counts[0, decision], value[decision], supported, posterior, tracking
            )
            movement = (scores - scores.gather(1, top)).masked_fill(~supported, 0)
            logits[decision] = prior + self.scale * movement
        return logits, value, (*next_state, counts, *extra_state)


def from_bundle(bundle, device="cpu"):
    root = Path(__file__).resolve().parent
    path = root / bundle["base"]
    if file_hash(path) != bundle["base_sha256"]:
        raise ValueError("counterfactual rollout base checkpoint hash mismatch")
    policy = teacher(1, device=device, path=path)
    style = None
    if "style" in bundle:
        from opponent_style import OpponentStyle

        stored = bundle["style"]
        style = OpponentStyle(stored["input_size"], stored["hidden_size"])
        style.load_state_dict(stored["state_dict"])
    head = BranchActionHead(
        graph=bundle["graph"],
        style_heads=style is not None,
        frequencies=bundle["head"].get("frequencies"),
    )
    if "table_offsets" in bundle["head"]:
        head.table_offsets = torch.empty_like(bundle["head"]["table_offsets"])
    head.load_state_dict(bundle["head"])
    policy.puffer_policy = BranchOutcomeCore(
        NativeSuitOrbitCore(policy.puffer_policy), head, bundle["scale"], bundle["margin"], style
    )
    policy._decision_features = _decision_features
    return policy.to(device).eval().requires_grad_(False)
