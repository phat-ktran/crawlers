with open('data/postprocess/segmented/train.txt', 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split('\t')
        if len(parts) != 4:
            continue  # Skip lines that don't have the expected format

        max_length = max(len(parts[1]), locals().get('max_length', 0))
        
    print(max_length)