"""Synthetic Manglish corruption: character repetition plus random duplication/deletion."""
import random
from config import P_REPEAT, P_DUPDEL, NOISE_SEED


def corrupt(text: str, rng: random.Random, p_rep: float = P_REPEAT, p_dd: float = P_DUPDEL) -> str:
    out = []
    for ch in text:
        if not ch.isalpha():
            out.append(ch)
            continue
        r = rng.random()
        if r < p_rep:
            out.append(ch * (1 + rng.randint(1, 3)))
        elif r < p_rep + p_dd:
            if rng.random() < 0.5:
                out.append(ch * 2)
            # else: deletion, append nothing
        else:
            out.append(ch)
    noisy = "".join(out)
    return noisy if noisy.strip() else text


def corrupt_all(sentences, seed: int = NOISE_SEED):
    """Deterministic per-sentence noise so every backbone sees the identical corrupted data."""
    return [corrupt(s, random.Random(seed * 1_000_003 + i)) for i, s in enumerate(sentences)]


if __name__ == "__main__":
    for s in ["povilla", "njan nale varum", "ente ammaye ariyo"]:
        print(s, "->", corrupt(s, random.Random(1)))
