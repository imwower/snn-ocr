"""Pure-Python CTC utilities: forward-backward loss and decoding helpers."""
from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

Logits = Sequence[Sequence[float]]
Indices = Sequence[int]

_NEG_INF = float("-inf")
SYMBOLS: Tuple[str, ...] = tuple(
    [""]  # blank at index 0
    + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("abcdefghijklmnopqrstuvwxyz")
    + list("0123456789")
    + [" ", ".", ",", "!", "?"]
)


def symbol_table() -> Tuple[str, ...]:
    """Expose the static symbol table (blank-inclusive)."""
    return SYMBOLS


def _log_sum_exp(a: float, b: float) -> float:
    if a == _NEG_INF:
        return b
    if b == _NEG_INF:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def _log_sum_exp_list(values: Iterable[float]) -> float:
    iterator = iter(values)
    try:
        acc = next(iterator)
    except StopIteration:
        return _NEG_INF
    for value in iterator:
        acc = _log_sum_exp(acc, value)
    return acc


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


def _ctc_forward(
    logits: Logits, target: Sequence[int], blank: int
) -> Tuple[List[List[float]], List[int], List[List[float]], float, float]:
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
            total = _log_sum_exp_list(terms)
            alpha[t][s] = log_probs[t][current_symbol] + total
    final_candidates = [alpha[-1][S - 1]]
    if S > 1:
        final_candidates.append(alpha[-1][S - 2])
    log_likelihood = _log_sum_exp_list(final_candidates)
    loss = -log_likelihood
    return log_probs, extended, alpha, loss, log_likelihood


def ctc_loss(logits: Logits, target: Sequence[int], blank: int = 0) -> float:
    """Compute the negative log-likelihood of target under the CTC objective."""
    _, _, _, loss, _ = _ctc_forward(logits, target, blank)
    return loss


def ctc_loss_with_grad(
    logits: Logits, target: Sequence[int], blank: int = 0
) -> Tuple[float, List[List[float]]]:
    """CTC loss and gradient w.r.t. logits."""
    log_probs, extended, alpha, loss, log_likelihood = _ctc_forward(
        logits, target, blank
    )
    T = len(log_probs)
    S = len(extended)
    num_classes = len(log_probs[0])
    beta = [[_NEG_INF for _ in range(S)] for _ in range(T)]
    beta[T - 1][S - 1] = 0.0
    if S > 1:
        beta[T - 1][S - 2] = 0.0
    for t in range(T - 2, -1, -1):
        for s in range(S):
            transitions: List[float] = []
            symbol_s = extended[s]
            # stay
            if beta[t + 1][s] != _NEG_INF:
                transitions.append(beta[t + 1][s] + log_probs[t + 1][symbol_s])
            # move to s+1
            if s + 1 < S and beta[t + 1][s + 1] != _NEG_INF:
                transitions.append(beta[t + 1][s + 1] + log_probs[t + 1][extended[s + 1]])
            # skip over blank
            if (
                s + 2 < S
                and extended[s] != blank
                and extended[s] != extended[s + 2]
                and beta[t + 1][s + 2] != _NEG_INF
            ):
                transitions.append(beta[t + 1][s + 2] + log_probs[t + 1][extended[s + 2]])
            beta[t][s] = _log_sum_exp_list(transitions) if transitions else _NEG_INF
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


def beam_search(logits: Logits, beam: int = 5, blank: int = 0) -> str:
    """CTC beam search decoding with prefix probabilities."""
    if beam <= 0:
        raise ValueError("beam must be positive")
    if not logits:
        return ""
    log_probs = [_log_softmax_row(frame) for frame in logits]
    beams: dict[Tuple[int, ...], Tuple[float, float]] = {(): (0.0, _NEG_INF)}
    for log_prob in log_probs:
        next_beams: dict[Tuple[int, ...], Tuple[float, float]] = {}
        for prefix, (pb, pn) in beams.items():
            log_p_blank = log_prob[blank]
            total_blank = _log_sum_exp(pb, pn)
            new_pb, new_pn = next_beams.get(prefix, (_NEG_INF, _NEG_INF))
            new_pb = _log_sum_exp(new_pb, total_blank + log_p_blank)
            next_beams[prefix] = (new_pb, new_pn)
            for symbol, log_p_symbol in enumerate(log_prob):
                if symbol == blank:
                    continue
                last = prefix[-1] if prefix else None
                if last == symbol:
                    # stay in same prefix (no blank between repeats)
                    same_pb, same_pn = next_beams.get(prefix, (_NEG_INF, _NEG_INF))
                    same_pn = _log_sum_exp(same_pn, pn + log_p_symbol)
                    next_beams[prefix] = (same_pb, same_pn)
                    # extend with blank gap
                    new_prefix = prefix + (symbol,)
                    new_pb_state, new_pn_state = next_beams.get(new_prefix, (_NEG_INF, _NEG_INF))
                    new_pn_state = _log_sum_exp(new_pn_state, pb + log_p_symbol)
                    next_beams[new_prefix] = (new_pb_state, new_pn_state)
                else:
                    new_prefix = prefix + (symbol,)
                    new_pb_state, new_pn_state = next_beams.get(new_prefix, (_NEG_INF, _NEG_INF))
                    merged = _log_sum_exp(pb + log_p_symbol, pn + log_p_symbol)
                    new_pn_state = _log_sum_exp(new_pn_state, merged)
                    next_beams[new_prefix] = (new_pb_state, new_pn_state)
        sorted_beams = sorted(
            next_beams.items(),
            key=lambda item: _log_sum_exp(item[1][0], item[1][1]),
            reverse=True,
        )
        beams = dict(sorted_beams[:beam])
    best_prefix = max(
        beams.items(),
        key=lambda item: _log_sum_exp(item[1][0], item[1][1]),
    )[0]
    return "".join(_index_to_symbol(idx, blank) for idx in best_prefix)


def _self_check() -> None:
    logits = [
        [2.0, 4.0, 1.0],
        [3.0, 1.0, 2.0],
        [2.5, 0.5, 3.5],
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


if __name__ == "__main__":
    _self_check()
    toy_logits = [
        [1.2, 2.5, 0.2],  # mostly 'A'
        [2.8, 0.7, 0.3],  # blank
        [0.9, 0.4, 3.0],  # mostly 'B'
    ]
    print("Greedy decode:", greedy_decode(toy_logits, blank=0))
    print("Beam search decode:", beam_search(toy_logits, beam=5, blank=0))
