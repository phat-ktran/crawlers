import torch
from Levenshtein import distance as levenshtein_distance


PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3

def token2str(tokens, mask, id_to_token, training=False):
    """
    Convert token IDs to string, ignoring PAD/SOS/EOS and masked positions.
    
    Args:
        tokens: list[int], token ids
        mask: list[int], same length, 1=valid, 0=ignore
        training: bool, whether we are decoding training targets (mask includes SOS)
    """
    result = []
    for idx, (token, m) in enumerate(zip(tokens, mask)):
        if m == 0:  # masked padding
            continue
        if token == PAD_ID:
            continue
        if training and idx == 0 and token == SOS_ID:  # skip <SOS> at first pos
            continue
        if token == EOS_ID:  # stop at EOS
            break
        result.append(id_to_token[token])
    return "".join(result)


def pad_y_shift_to_string(pad_y_shift, mask, id_to_token, training=True):
    """
    Convert batch of shifted target sequences into strings.
    Args:
        pad_y_shift: (batch_size, seq_len)
        mask: (batch_size, seq_len)
        training: bool
    """
    decoded_strings = []
    for sequence, seq_mask in zip(pad_y_shift, mask):
        decoded_strings.append(
            token2str(sequence.tolist(), seq_mask.tolist(), id_to_token, training=training)
        )
    return decoded_strings


def greedy_decoding(logits, mask, id_to_token):
    """
    Greedy decode from logits into strings, using inference mask.
    Args:
        logits: (batch_size, seq_len, vocab_size)
        mask:   (batch_size, seq_len)
    """
    decoded_sequences = []
    token_ids = torch.argmax(logits, dim=-1)  # (batch_size, seq_len)
    for sequence, seq_mask in zip(token_ids, mask):
        decoded_sequences.append(
            token2str(sequence.tolist(), seq_mask.tolist(), id_to_token, training=False)
        )
    return decoded_sequences

def compute_avg_cer(decoded_sequences, ground_truths):
    cer_list = []

    for decoded, gt in zip(decoded_sequences, ground_truths):
        distance = levenshtein_distance(decoded[:len(gt)], gt)
        cer = distance / len(gt) if len(gt) > 0 else 0.0
        cer_list.append(cer)

    return sum(cer_list) / len(cer_list) if cer_list else 0.0
