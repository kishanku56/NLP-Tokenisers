from collections import defaultdict, Counter
from datetime import datetime
import argparse
import heapq
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_token = False

def build_trie(vocab):
    root = TrieNode()
    for token in vocab:
        node = root
        for ch in token:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_token = True
    return root

def tokenize_trie(word, trie):
    tokens, i = [], 0
    while i < len(word):
        node = trie
        match = None
        j = i
        while j < len(word) and word[j] in node.children:
            node = node.children[word[j]]
            j += 1
            if node.is_token:
                match = j
        if match: 
            sub = word[i:match]
            tokens.append(sub)
            i = match
        else:  
            tokens.append(word[i:i+1])
            i += 1
    return tokens



class Node:
    def __init__(self, token):
        self.token = token
        self.prev = None
        self.next = None
        self.stale=False        # for bypassing nodes in heap


def encode_word(word):
    return list(word.encode("utf-8"))


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
    reserve_words=["<pad>", "<unk>", "<s>", "</s>"]

    if not corpus_words:
        return merges, reserve_words,pair2id

    word_freqs = Counter()
    word_freqs[corpus_words]=1


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
    vocab.extend(reserve_words)
    vocab.extend([merge_rule_to_string(pair,id2bytes) for pair in merges])
    #print("Vocab:", vocab)

    return merges, vocab, pair2id



def vocab_to_bytes(vocab):
    import re
    byte_tokens_all = []
    for text in vocab:
        tokens = re.findall(r"<\d+>|[^<>]+", text)
        # Step 2: encode each token to bytes
        byte_stream = bytearray()
        for tok in tokens:
            if tok.startswith("<") and tok.endswith(">") and tok[1:-1].isdigit():
                num = int(tok[1:-1])
                if 0 <= num <= 255:
                    byte_stream+=bytes([num])  # single byte
                else:
                    # number >255 → encode whole '<number>' as UTF-8
                    byte_stream+=tok.encode("utf-8")
            else:
                byte_stream+=tok.encode("utf-8")
        
        final_bytes = bytes(byte_stream)
        byte_tokens_all.append(final_bytes)
    
    return byte_tokens_all



def tokenize(vocab,text):
    byte_vocab = vocab_to_bytes(vocab)
    trie = build_trie(byte_vocab) 
    text_bytes = text.encode('utf-8') 
    return tokenize_trie(text_bytes, trie)


def detokenize(tokens):
    merged_bytes = b''.join(tokens)
    text=merged_bytes.decode("utf-8")
    return text.replace("\u2581"," ")


def save_vocab(vocab, rollno):
    fname = f"{rollno}_assignment2_sp_vocab_{len(vocab)}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in vocab:
            f.write(str(tok) + "\n")

def save_tokens(tokens, rollno):
    fname = f"{rollno}_assignment2_sp_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in tokens:
            f.write(str(tok) + "\n")

def save_detokenized(text, rollno):
    fname = f"{rollno}_assignment2_sp_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)

def load_training_data(train_path):
    with open(train_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def preprocess_text(text):
    text = unicodedata.normalize('NFKC', text)
    text = text.lower()
    text = text.replace(' ', '\u2581')
    return text

def decode_token(token):
    try:
        return token.decode('utf-8')
    except UnicodeDecodeError:
        return ''.join(f'<{b}>' for b in token)


if __name__ == "__main__":  
    import multiprocessing as mp
    from collections import OrderedDict
    import unicodedata
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
    train_text = preprocess_text(train_text)

    merges, vocab, pair2id = train_bpe(train_text, num_merges=args.vocab_size)
    save_vocab(vocab, rollno)
    
    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = f.read()
    
    sample_text=preprocess_text(sample_text)
    tokens = tokenize(vocab,sample_text)
    decoded_tokens = [decode_token(tok) for tok in tokens]
    save_tokens(decoded_tokens, rollno)

    detok_text = detokenize(tokens)
    save_detokenized(detok_text, rollno)
