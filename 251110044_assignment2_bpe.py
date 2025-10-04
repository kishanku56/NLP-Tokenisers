class Node:
    def __init__(self, token):
        self.token = token
        self.prev = None
        self.next = None
        self.stale=False        # for bypassing nodes in heap


def encode_word(word):
    return [start_marker_id] + list(word.encode("utf-8"))


def merge_rule_to_string(pair, id2bytes):
    a, b = pair
    bytes_seq = id2bytes[a] + id2bytes[b]
    try:
        return bytes_seq.decode("utf-8")
    except UnicodeDecodeError:
        return "".join(f"<{b}>" for b in bytes_seq) # This happens for chars which have multi byte representation and we get partial merging, For e.g in Hindi


def build_linked_list(byte_ids):
    nodes = [Node(tid) for tid in byte_ids]
    for i in range(1, len(nodes)):
        nodes[i].prev = nodes[i-1]
        nodes[i-1].next = nodes[i]
    return nodes


def pair_as_bytes(pair):
        a, b = pair
        return id2bytes[a] + id2bytes[b]


def train_bpe(corpus_words, num_merges=1000):
    global next_id, id2bytes
    pair2id = {}
    merges = []

    word_freqs = Counter()
    for w in corpus_words:
        word_freqs[w] += 1


    # Build linked lists for each word
    corpus = {w: build_linked_list(encode_word(w)) for w in word_freqs}

    # Pair counts and positions
    pair_counts = Counter()
    pair_positions = defaultdict(list)
    for w, nodes in corpus.items():
        node = nodes[0]
        while node and node.next:
            pair = (node.token, node.next.token)
            pair_counts[pair] += word_freqs[w]
            pair_positions[pair].append((node, word_freqs[w]))
            node = node.next

    # Build max-heap by using negative count in a min heap
    heap = [(-count, pair_as_bytes(pair), pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)

    while(len(merges)<num_merges-4):    

        # Pop most frequent valid pair until heap becomes empty 
        while heap:
            neg_count, _, best_pair = heapq.heappop(heap)
            if pair_counts.get(best_pair, 0) == -neg_count:     # This is required since heap might contain stale data
                break
        else:
            break


        # Assign new token ID
        if best_pair not in pair2id:
            pair2id[best_pair] = next_id
            id2bytes[next_id] = id2bytes[best_pair[0]] + id2bytes[best_pair[1]]
            next_id += 1
        new_id = pair2id[best_pair]

        # Merge occurrences
        positions = pair_positions[best_pair].copy()    # we have to iterate over a copy since we will modify the original list

        updated_keys=set()
        merged_at_least_once = True  # mark that we did a merge
        for node, freq in positions:
            if node.stale or not node.next:         # node.next is initially not None but can become due to merging (a,(bc))
                continue

            if (node.token, node.next.token) != best_pair:  # due to merging in the neighbourhood
                continue
            
            # Mark old adjacent pairs as stale
            for neighbor in [node.prev, node.next]:
                if neighbor and neighbor.next:
                    old_pair = (neighbor.token, neighbor.next.token)
                    pair_counts[old_pair] -= freq       
            
            node.next.stale=True    # suppose text is "differ" if current pair is ff then fe becomes stale(stale nodes are not processed)
                                    # prev node is not marked stale(it will point to the merged node)

            # Merge nodes
            node.token = new_id
            node.next = node.next.next  #suppose text is "differ" next of ff becomes er(fe is bypassed)
            if node.next:
                node.next.prev = node   # er is linked back to ff(fe is bypassed)



            # Update new adjacent pairs
            for neighbor in [node.prev, node]:
                if neighbor and neighbor.next:
                    pair = (neighbor.token, neighbor.next.token)
                    pair_counts[pair] += freq       
                    pair_positions[pair].append((neighbor, freq))
                    updated_keys.add(pair)


            merged_at_least_once = True  # mark that we did a merge
        
        for key in updated_keys:
            heapq.heappush(heap, (-pair_counts[key], pair_as_bytes(key), key))
        
        pair_positions[best_pair]=[]
        pair_counts[best_pair]=0
        
        if merged_at_least_once:
            merges.append(best_pair)

    vocab=[]
    vocab.extend(["<pad>", "<unk>", "<s>", "</s>"])
    vocab.extend([merge_rule_to_string(pair,id2bytes) for pair in merges])
    #print("Vocab:", vocab)

    return merges, vocab, pair2id


def merge_once(word, pair2id, pair2rank):
    pairs = [(i, (word[i], word[i+1])) for i in range(len(word)-1)]
    ranked = [(pair2rank[p], i, p) for i, p in pairs if p in pair2rank]
    if not ranked:
        return word, False
    _, idx, pair = min(ranked, key=lambda x: x[0])
    return word[:idx] + [pair2id[pair]] + word[idx+2:], True


def tokenize_word(args):
    w, pair2id, pair2rank, id2bytes, start_marker, start_marker_id=args
    # encode word to bytes
    word_bytes = [start_marker_id] + list(w.encode("utf-8"))

    # repeatedly merge until no change
    changed = True
    while changed:
        word_bytes, changed = merge_once(word_bytes, pair2id, pair2rank)

    # decode tokens
    tokens = []
    for tok in word_bytes:
        if tok == start_marker_id:
            tokens.append(start_marker)
        else:
            try:
                tokens.append(id2bytes[tok].decode("utf-8"))
            except UnicodeDecodeError:
                tokens.append(list(id2bytes[tok]))
    return w, tokens

def tokenize(text, merges, pair2id):
    from multiprocessing import Pool

    words = text.split()

    # keep order but deduplicate
    # avoids processing the same word again and again
    unique_words = list(OrderedDict.fromkeys(words))        # order is retained for logging consistency, will work with a normal set also

    # build pair2rank once
    pair2rank = {pair: i for i, pair in enumerate(merges)}

    num_workers = mp.cpu_count()

    with Pool(processes=num_workers) as pool:
        results = pool.imap_unordered(
            tokenize_word,
            [(w, pair2id, pair2rank, id2bytes, start_marker, start_marker_id)
             for w in unique_words],
            chunksize=100   
        )

        word2tokens = dict(results)

    # reconstruct original sequence
    final_tokens = []
    for w in words:
        final_tokens.extend(word2tokens[w])

    return final_tokens


def detokenize(tokens):
    out_bytes = bytearray()

    for tok in tokens:
        if isinstance(tok, str):
            word=tok
            if(word[0]==start_marker):
                word=" "+word[1:]
            out_bytes.extend(word.encode("utf-8"))
        elif isinstance(tok, list):
            out_bytes.extend(tok)

    text = out_bytes.decode("utf-8")
    if(text.startswith(" ")):
        text = text[1:]
    return text


def save_vocab(vocab, rollno):
    fname = f"{rollno}_assignment2_bpe_vocab_{len(vocab)}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in vocab:
            f.write(str(tok) + "\n")

def save_tokens(tokens, rollno):
    fname = f"{rollno}_assignment2_bpe_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in tokens:
            f.write(str(tok) + "\n")

def save_detokenized(text, rollno):
    fname = f"{rollno}_assignment2_bpe_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)

def load_training_data(train_path):
    with open(train_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text.split()


if __name__ == "__main__":  
    from collections import defaultdict, Counter
    from datetime import datetime
    import argparse
    import heapq
    import multiprocessing as mp
    from collections import OrderedDict
    start_marker = '_'
    start_marker_id = 256  # token ID for start marker, we dont map it to underscore byte value, this way if text contains underscore it will be processed normally
    id2bytes = {i: bytes([i]) for i in range(256)}
    id2bytes[start_marker_id] = start_marker.encode("utf-8")
    next_id = start_marker_id + 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()

    rollno = "251110044"
    train_text = load_training_data(args.train)

    merges, vocab, pair2id = train_bpe(train_text, num_merges=args.vocab_size)
    save_vocab(vocab, rollno)
    
    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = f.read()

    tokens = tokenize(sample_text, merges, pair2id)
    save_tokens(tokens, rollno)
    
    detok_text = detokenize(tokens)
    save_detokenized(detok_text, rollno)


    