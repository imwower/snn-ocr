"""Pure-Python CTC utilities: forward-backward loss and decoding helpers."""
from __future__ import annotations

import math
from random import Random
from typing import Dict, List, Sequence, Tuple

Logits = Sequence[Sequence[float]]
Indices = Sequence[int]

_NEG_INF = float("-inf")
SYMBOLS: Tuple[str, ...] = tuple(
    [""]  # blank at index 0
    + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("abcdefghijklmnopqrstuvwxyz")
    + list("0123456789")
    + [" ", ".", ",", "!", "?", "\n"]
)


def symbol_table() -> Tuple[str, ...]:
    """Expose the static symbol table (blank-inclusive)."""
    return SYMBOLS


def log_sum_exp(*values: float) -> float:
    """Stable log-sum-exp that accepts any number of values."""
    filtered = [value for value in values if value != _NEG_INF]
    if not filtered:
        return _NEG_INF
    maximum = max(filtered)
    if math.isinf(maximum):
        return maximum
    total = sum(math.exp(value - maximum) for value in filtered)
    return maximum + math.log(total)


def _log_softmax_row(row: Sequence[float]) -> List[float]:
    maximum = max(row)
    shifted = [value - maximum for value in row]
    exps = [math.exp(value) for value in shifted]
    total = sum(exps)
    return [shifted[i] - math.log(total) for i in range(len(row))]


def _extend_with_blanks(target: Sequence[int], blank: int) -> List[int]:
    extended = [blank]
    for symbol in target:
        extended.append(symbol)
        extended.append(blank)
    return extended


def _forward_backward(
    logits: Logits, target: Sequence[int], blank: int
) -> Tuple[
    List[List[float]],
    List[int],
    List[List[float]],
    List[List[float]],
    float,
    float,
]:
    """Shared forward-backward core returning intermediate tensors."""
    if not logits:
        raise ValueError("logits must be non-empty")
    log_probs = [_log_softmax_row(frame) for frame in logits]
    T = len(log_probs)
    extended = _extend_with_blanks(target, blank)
    S = len(extended)
    alpha = [[_NEG_INF for _ in range(S)] for _ in range(T)]
    alpha[0][0] = log_probs[0][blank]
    if S > 1:
        alpha[0][1] = log_probs[0][extended[1]]
    for t in range(1, T):
        for s in range(S):
            current_symbol = extended[s]
            terms = [alpha[t - 1][s]]
            if s - 1 >= 0:
                terms.append(alpha[t - 1][s - 1])
            if (
                s - 2 >= 0
                and current_symbol != blank
                and current_symbol != extended[s - 2]
            ):
                terms.append(alpha[t - 1][s - 2])
            total = log_sum_exp(*terms)
            alpha[t][s] = log_probs[t][current_symbol] + total
    final_candidates = [alpha[-1][S - 1]]
    if S > 1:
        final_candidates.append(alpha[-1][S - 2])
    log_likelihood = log_sum_exp(*final_candidates)
    loss = -log_likelihood

    beta = [[_NEG_INF for _ in range(S)] for _ in range(T)]
    beta[T - 1][S - 1] = 0.0
    if S > 1:
        beta[T - 1][S - 2] = 0.0
    for t in range(T - 2, -1, -1):
        for s in range(S):
            transitions: List[float] = []
            symbol_s = extended[s]
            if beta[t + 1][s] != _NEG_INF:
                transitions.append(beta[t + 1][s] + log_probs[t + 1][symbol_s])
            if s + 1 < S and beta[t + 1][s + 1] != _NEG_INF:
                transitions.append(beta[t + 1][s + 1] + log_probs[t + 1][extended[s + 1]])
            if (
                s + 2 < S
                and extended[s] != blank
                and extended[s] != extended[s + 2]
                and beta[t + 1][s + 2] != _NEG_INF
            ):
                transitions.append(beta[t + 1][s + 2] + log_probs[t + 1][extended[s + 2]])
            beta[t][s] = log_sum_exp(*transitions) if transitions else _NEG_INF
    return log_probs, extended, alpha, beta, loss, log_likelihood


def ctc_loss(logits: Logits, target: Sequence[int], blank: int = 0) -> float:
    """Compute the negative log-likelihood of target under the CTC objective."""
    _, _, _, _, loss, _ = _forward_backward(logits, target, blank)
    return loss


def ctc_loss_with_grad(
    logits: Logits, target: Sequence[int], blank: int = 0
) -> Tuple[float, List[List[float]]]:
    """CTC loss and gradient w.r.t. logits."""
    log_probs, extended, alpha, beta, loss, log_likelihood = _forward_backward(
        logits, target, blank
    )
    T = len(log_probs)
    S = len(extended)
    num_classes = len(log_probs[0])
    probs = [[math.exp(value) for value in row] for row in log_probs]
    grad = [[probs[t][k] for k in range(num_classes)] for t in range(T)]
    for t in range(T):
        posterior_by_symbol = [0.0 for _ in range(num_classes)]
        for s in range(S):
            symbol = extended[s]
            if symbol >= num_classes:
                continue
            alpha_ts = alpha[t][s]
            beta_ts = beta[t][s]
            if alpha_ts == _NEG_INF or beta_ts == _NEG_INF:
                continue
            posterior = math.exp(alpha_ts + beta_ts - log_likelihood)
            posterior_by_symbol[symbol] += posterior
        for k in range(num_classes):
            grad[t][k] -= posterior_by_symbol[k]
    return loss, grad


def _collapse_repeats(indices: Sequence[int], blank: int) -> List[int]:
    collapsed: List[int] = []
    prev: int | None = None
    for idx in indices:
        if idx == blank:
            prev = None
            continue
        if idx == prev:
            continue
        collapsed.append(idx)
        prev = idx
    return collapsed


def _index_to_symbol(index: int, blank: int) -> str:
    if index == blank:
        return ""
    if index < len(SYMBOLS):
        return SYMBOLS[index]
    return str(index)


def _best_alignment_path(
    logits: Logits, target: Sequence[int], blank: int
) -> List[int]:
    """Return the most likely alignment path (used for __main__ demo)."""
    _, extended, alpha, beta, _, _ = _forward_backward(logits, target, blank)
    alignment: List[int] = []
    if not alpha:
        return alignment
    S = len(extended)
    for t in range(len(alpha)):
        best_state = 0
        best_score = _NEG_INF
        for s in range(S):
            score = alpha[t][s] + beta[t][s]
            if score > best_score:
                best_score = score
                best_state = s
        alignment.append(extended[best_state])
    return alignment


def best_alignment_path(logits: Logits, target: Sequence[int], blank: int = 0) -> List[int]:
    """Public helper that exposes the most likely alignment path for visualization."""
    return _best_alignment_path(logits, target, blank)


def greedy_decode(logits: Logits, blank: int = 0) -> str:
    """Return a collapsed character sequence using greedy decoding."""
    if not logits:
        return ""
    argmax_indices: List[int] = []
    for frame in logits:
        best_idx = max(range(len(frame)), key=lambda i: frame[i])
        argmax_indices.append(best_idx)
    collapsed = _collapse_repeats(argmax_indices, blank)
    return "".join(_index_to_symbol(idx, blank) for idx in collapsed)


def beam_search(
    logits: Logits,
    beam: int = 5,
    blank: int = 0,
    len_norm: bool = True,
    ins_penalty: float = 0.0,
) -> str:
    """CTC prefix beam search with blank/non-blank states."""
    if beam <= 0:
        raise ValueError("beam must be positive")
    if not logits:
        return ""
    log_probs = [_log_softmax_row(frame) for frame in logits]
    beams: Dict[Tuple[int, ...], Tuple[float, float]] = {(): (0.0, _NEG_INF)}
    for log_prob in log_probs:
        next_beams: Dict[Tuple[int, ...], Tuple[float, float]] = {}
        for prefix, (pb, pnb) in beams.items():
            total = log_sum_exp(pb, pnb)
            blank_logp = log_prob[blank]
            cur_pb, cur_pnb = next_beams.get(prefix, (_NEG_INF, _NEG_INF))
            cur_pb = log_sum_exp(cur_pb, total + blank_logp)
            next_beams[prefix] = (cur_pb, cur_pnb)
            for symbol, log_p_symbol in enumerate(log_prob):
                if symbol == blank:
                    continue
                last = prefix[-1] if prefix else None
                if last == symbol:
                    stay_pb, stay_pnb = next_beams.get(prefix, (_NEG_INF, _NEG_INF))
                    stay_pnb = log_sum_exp(stay_pnb, pnb + log_p_symbol)
                    next_beams[prefix] = (stay_pb, stay_pnb)
                    new_prefix = prefix + (symbol,)
                    nb_pb, nb_pnb = next_beams.get(new_prefix, (_NEG_INF, _NEG_INF))
                    nb_pnb = log_sum_exp(nb_pnb, pb + log_p_symbol)
                    next_beams[new_prefix] = (nb_pb, nb_pnb)
                else:
                    new_prefix = prefix + (symbol,)
                    nb_pb, nb_pnb = next_beams.get(new_prefix, (_NEG_INF, _NEG_INF))
                    nb_pnb = log_sum_exp(nb_pnb, total + log_p_symbol)
                    next_beams[new_prefix] = (nb_pb, nb_pnb)
        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda item: log_sum_exp(item[1][0], item[1][1]),
                reverse=True,
            )[:beam]
        )

    def _final_score(entry: Tuple[Tuple[int, ...], Tuple[float, float]]) -> float:
        prefix, (pb, pnb) = entry
        length = len(prefix)
        base = log_sum_exp(pb, pnb) - ins_penalty * length
        if len_norm and length > 0:
            return base / length
        return base

    best_prefix = max(beams.items(), key=_final_score)[0]
    return "".join(_index_to_symbol(idx, blank) for idx in best_prefix)


def _self_check() -> None:
    logits = [
        [2.0, 4.0, 1.0],
        [2.5, 0.5, 3.5],
        [3.0, 1.0, 2.0],
    ]
    target = [1, 2]
    loss = ctc_loss(logits, target, blank=0)
    assert loss > 0
    loss2, grad = ctc_loss_with_grad(logits, target, blank=0)
    assert abs(loss - loss2) < 1e-6
    assert len(grad) == len(logits)
    greedy = greedy_decode(logits, blank=0)
    beam = beam_search(logits, beam=3, blank=0)
    assert greedy == beam


def _format_target(target: Sequence[int], blank: int) -> str:
    if not target:
        return "<empty>"
    return "".join(_index_to_symbol(idx, blank) for idx in target)


def _demo_case(name: str, logits: List[List[float]], target: Sequence[int]) -> None:
    loss_value = ctc_loss(logits, target, blank=0)
    greedy = greedy_decode(logits, blank=0)
    beam = beam_search(
        logits,
        beam=8,
        blank=0,
        len_norm=True,
        ins_penalty=0.05,
    )
    print(f"[{name}] target={_format_target(target, blank=0)} loss={loss_value:.4f}")
    print(f"  greedy={greedy!r}")
    print(f"  beam  ={beam!r}")


if __name__ == "__main__":
    _self_check()
    empty_target_logits = [
        [3.5, 0.1, 0.2],
        [3.0, 0.5, 0.1],
        [2.8, 0.2, 0.2],
    ]
    repeated_logits = [
        [0.2, 2.5, 0.1],  # prefer 'A'
        [0.1, 2.2, 0.3],  # prefer 'A'
        [2.4, 0.2, 0.4],  # encourage blank separation
        [0.2, 2.4, 0.3],  # second 'A'
    ]
    repeated_target = [1, 1]  # "AA"
    rng_logits = []
    rng = Random(13)
    for _ in range(128):
        frame = [rng.uniform(-0.5, 0.5) for _ in range(5)]
        frame[0] += 0.3  # keep blank competitive
        rng_logits.append(frame)
    rng_target = [1, 2, 3]  # "A", "B", "C"
    _demo_case("empty-target", empty_target_logits, [])
    _demo_case("double-letter", repeated_logits, repeated_target)
    _demo_case("random-noise", rng_logits, rng_target)
