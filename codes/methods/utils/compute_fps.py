with open('data/postprocess/train_aug.txt', 'r', encoding='utf-8') as file:
    tp_count = 0
    tn_count = 0
    fp_count = 0
    fn_count = 0

    for line in file:
        parts = line.strip().split('\t')
        if len(parts) != 4:
            continue  # Skip lines that don't have the expected format

        binary_flag = parts[2].split(',')
        B_prime = ['1' if c1 == c2 else '0' for c1, c2 in zip(parts[0], parts[1])]

        for idx, flag in enumerate(binary_flag):
            if idx < len(B_prime):
                if flag == '1' and B_prime[idx] == '1':
                    tp_count += 1
                elif flag == '0' and B_prime[idx] == '0':
                    tn_count += 1
                elif flag == '1' and B_prime[idx] == '0':
                    fp_count += 1
                elif flag == '0' and B_prime[idx] == '1':
                    fn_count += 1

print(f"TP: {tp_count}, TN: {tn_count}, FP: {fp_count}, FN: {fn_count}")

precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")