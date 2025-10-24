"""Pure-Python CTC utilities: forward-backward loss and decoding helpers."""
from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

Logits = Sequence[Sequence[float]]
Indices = Sequence[int]

_NEG_INF = float("-inf")
_SYMBOL_TABLE: Sequence[str] = (
    [""]  # placeholder for blank at index 0
    + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("abcdefghijklmnopqrstuvwxyz")
    + list("0123456789")
    + [" "]
)


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


def ctc_loss(logits: Logits, target: Sequence[int], blank: int = 0) -> float:
    """Compute the negative log-likelihood of target under the CTC objective."""
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
            total = alpha[t - 1][s]
            if s - 1 >= 0:
                total = _log_sum_exp(total, alpha[t - 1][s - 1])
            if (
                s - 2 >= 0
                and current_symbol != blank
                and current_symbol != extended[s - 2]
            ):
                total = _log_sum_exp(total, alpha[t - 1][s - 2])
            alpha[t][s] = log_probs[t][current_symbol] + total
    final_candidates = [alpha[-1][S - 1]]
    if S > 1:
        final_candidates.append(alpha[-1][S - 2])
    return -_log_sum_exp_list(final_candidates)


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
    if index < len(_SYMBOL_TABLE):
        return _SYMBOL_TABLE[index]
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
