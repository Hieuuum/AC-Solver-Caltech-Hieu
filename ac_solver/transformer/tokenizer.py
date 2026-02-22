"""
Tokenizer for converting balanced presentations between the PPO encoding
(integers 1, -1, 2, -2, 0) and the transformer token vocabulary (0-5).

Token vocabulary (6 tokens):
    0: x      (PPO value: 1)
    1: x^{-1} (PPO value: -1)
    2: y      (PPO value: 2)
    3: y^{-1} (PPO value: -2)
    4: SEP    (separator between r0 and r1)
    5: EOS    (end of sequence)
"""

import numpy as np

# Token constants
TOKEN_X = 0
TOKEN_X_INV = 1
TOKEN_Y = 2
TOKEN_Y_INV = 3
TOKEN_SEP = 4
TOKEN_EOS = 5
VOCAB_SIZE = 6

# Mapping from PPO encoding to transformer tokens
_PPO_TO_TOKEN = {1: TOKEN_X, -1: TOKEN_X_INV, 2: TOKEN_Y, -2: TOKEN_Y_INV}

# Mapping from transformer tokens to PPO encoding
_TOKEN_TO_PPO = {TOKEN_X: 1, TOKEN_X_INV: -1, TOKEN_Y: 2, TOKEN_Y_INV: -2}


def presentation_to_tokens(presentation, max_relator_length=None):
    """Convert a numpy presentation array to a transformer token sequence.

    The presentation is a 1D numpy array of length 2 * max_relator_length,
    where the first half is r0 (zero-padded) and the second half is r1 (zero-padded).

    The output token sequence is: [r0_tokens..., SEP, r1_tokens..., EOS]

    Parameters:
        presentation: numpy array or list of ints (PPO encoding)
        max_relator_length: if None, inferred as len(presentation) // 2

    Returns:
        list of ints (token IDs 0-5)
    """
    if isinstance(presentation, list):
        presentation = np.array(presentation, dtype=np.int8)

    if max_relator_length is None:
        max_relator_length = len(presentation) // 2

    r0 = presentation[:max_relator_length]
    r1 = presentation[max_relator_length : 2 * max_relator_length]

    r0_nonzero = r0[r0 != 0]
    r1_nonzero = r1[r1 != 0]

    tokens = []
    for val in r0_nonzero:
        tokens.append(_PPO_TO_TOKEN[int(val)])
    tokens.append(TOKEN_SEP)
    for val in r1_nonzero:
        tokens.append(_PPO_TO_TOKEN[int(val)])
    tokens.append(TOKEN_EOS)

    return tokens


def tokens_to_presentation(tokens, max_relator_length):
    """Convert a transformer token sequence back to a numpy presentation array.

    The token sequence should be: [r0_tokens..., SEP, r1_tokens..., EOS]

    Parameters:
        tokens: list of ints (token IDs 0-5)
        max_relator_length: the max relator length for zero-padding

    Returns:
        numpy array of dtype int8, length 2 * max_relator_length
    """
    # Find the separator position
    sep_idx = tokens.index(TOKEN_SEP)

    r0_tokens = tokens[:sep_idx]
    # r1 goes from after SEP to before EOS (or end if no EOS)
    if TOKEN_EOS in tokens[sep_idx + 1 :]:
        eos_idx = tokens.index(TOKEN_EOS, sep_idx + 1)
        r1_tokens = tokens[sep_idx + 1 : eos_idx]
    else:
        r1_tokens = tokens[sep_idx + 1 :]

    r0_ppo = [_TOKEN_TO_PPO[t] for t in r0_tokens]
    r1_ppo = [_TOKEN_TO_PPO[t] for t in r1_tokens]

    presentation = np.zeros(2 * max_relator_length, dtype=np.int8)
    presentation[: len(r0_ppo)] = r0_ppo
    presentation[max_relator_length : max_relator_length + len(r1_ppo)] = r1_ppo

    return presentation
