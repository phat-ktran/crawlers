import torch
from Levenshtein import distance as levenshtein_distance


PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3


def token2str(tokens, id_to_token):
    return "".join(
        id_to_token[token] for token in tokens if token not in {PAD_ID, SOS_ID, EOS_ID}
    )


def pad_y_shift_to_string(pad_y_shift, id_to_token):
    decoded_strings = []
    for sequence in pad_y_shift:
        decoded_strings.append(token2str(sequence.tolist(), id_to_token))
    return decoded_strings


def greedy_decoding(logits, id_to_token):
    decoded_sequences = []
    token_ids = torch.argmax(
        logits, dim=-1
    )  # Get the token IDs with the highest probability
    for sequence in token_ids:
        decoded_sequences.append(token2str(sequence.tolist(), id_to_token))
    return decoded_sequences


def compute_avg_cer(decoded_sequences, ground_truths):
    cer_list = []

    for decoded, gt in zip(decoded_sequences, ground_truths):
        distance = levenshtein_distance(decoded, gt)
        cer = distance / len(gt) if len(gt) > 0 else 0.0
        cer_list.append(cer)

    return sum(cer_list) / len(cer_list) if cer_list else 0.0
