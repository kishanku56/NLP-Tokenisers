

class Node:
    def __init__(self, token):
        self.token = token
        self.prev = None
        self.next = None
        self.stale = False

def build_linked_list(tokens):
    nodes = [Node(t) for t in tokens]
    for i in range(1, len(nodes)):
        nodes[i].prev = nodes[i-1]
        nodes[i-1].next = nodes[i]
    return nodes


def compute_deltaL(f_new, f_a, f_b, N):
    if f_new == 0 or N == 0:
        return float("-inf")
    
    delta_L=(f_new - (f_a + f_b)) * math.log(f_new / N)
    return delta_L


import math


def train_wordpiece(corpus_words, vocab_size=100, unk="[UNK]"):

    word_freqs = Counter()
    for w in corpus_words:
        word_freqs[w] += 1



    vocab = []
    vocab.extend(["<pad>", "<unk>", "<s>", "</s>"])
    for w in word_freqs:
        for ch in w:
            if ch not in vocab:
                vocab.append(ch)


    # Build linked lists for each word
    corpus = {w: build_linked_list(list(w)) for w in word_freqs}

    # Token counts - store frequency of each char across the corpus
    token_counts = Counter()
    total_tokens_count = 0
    for w, freq in word_freqs.items():
        for t in w:
            token_counts[t] += freq
            total_tokens_count += freq

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
    
    while len(vocab) < vocab_size:
        #print(len(vocab))        
        delta_values=Counter()
        for pair, f_new in pair_counts.items():
            if f_new==0:
                continue
            f_a = token_counts[pair[0]]
            f_b = token_counts[pair[1]]
            deltaL = compute_deltaL(f_new, f_a, f_b, total_tokens_count)
            delta_values[pair]=deltaL
        
        if(len(delta_values)==0):
            break

        # get pair with max deltaL, tie break by lexicographic ordering
        best_pair = max(delta_values, key=lambda p: (delta_values[p], p))
        best_deltaL = delta_values[best_pair]

        #print(best_pair, best_deltaL)

        count_decreased_by_this_merge=0

        new_token = "".join(best_pair)   
        '''
        if new_token in vocab:
            print("Token already exists")
            print(pair_counts[last_pair])
            print(vocab)
            input()
        '''
        vocab.append(new_token)

        # Merge occurrences
        positions = pair_positions[best_pair].copy()

        updated_pairs=set()

        nc=Counter()
        for node, freq in positions:
            if node.stale:          # due to merging in the neighbourhood
                continue

            if not node.next:       # if a larger string is selected first. (e,ii) followed by (e,i)
                continue
            
            if (node.token, node.next.token) != best_pair:  # due to merging in the neighbourhood
                continue

            # Update token counts
            token_counts[node.token] -= freq
            token_counts[node.next.token] -= freq
            token_counts[new_token] += freq

            
            # Mark old adjacent pairs as stale
            for neighbor in [node.prev, node.next]:
                if neighbor and neighbor.next:
                    old_pair = (neighbor.token, neighbor.next.token)
                    pair_counts[old_pair] -= freq       
            
            node.next.stale=True    # suppose text is "differ" if current pair is ff then fe becomes stale(stale nodes are not processed)
                                    # prev node is not marked stale(it will point to the merged node, must not be done)

            # Merge nodes
            node.token = new_token
            node.next = node.next.next  #suppose text is "differ" next of ff becomes er(fe is bypassed)
            if node.next:
                node.next.prev = node   # er is linked back to ff(fe is bypassed)



            # Update new adjacent pairs
            for neighbor in [node.prev, node]:
                if neighbor and neighbor.next:
                    pair = (neighbor.token, neighbor.next.token)
                    pair_counts[pair] += freq
                    #print(pair)       
                    pair_positions[pair].append((neighbor, freq))
                    updated_pairs.add(pair)            
            total_tokens_count-=freq

        del pair_positions[best_pair]
        del pair_counts[best_pair]

    return vocab





from multiprocessing import Pool, cpu_count

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

def wordpiece_tokenize_trie(word, trie, unk="[UNK]"):
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
            tokens.append(sub if i == 0 else "##" + sub)
            i = match
        else:  
            tokens.append(unk)
            i += 1
    return tokens


def tokenize_chunk(words, vocab, unk_token="[UNK]"):
    trie = build_trie(vocab) 
    tokens = []
    for w in words:
        tokens.extend(wordpiece_tokenize_trie(w, trie, unk_token))
    return tokens

def tokenize_text(text, vocab, unk_token="[UNK]"):
    words = text.strip().split()
    n = len(words)

    chunk_size = 1000
    chunks = [words[i:i+chunk_size] for i in range(0, n, chunk_size)]

    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(tokenize_chunk, [(chunk, vocab, unk_token) for chunk in chunks])

    return [tok for chunk_tokens in results for tok in chunk_tokens]


def detokenize(tokens):
    words = []
    current_word = ""

    for tok in tokens:
        if tok.startswith("##"):  # continuation of previous word
            current_word += tok[2:]
        else:
            if current_word:       
                words.append(current_word)
            current_word = tok
    if current_word:        #last word
        words.append(current_word)

    return " ".join(words)
    

def load_training_data(train_path):
    with open(train_path, "r", encoding="utf-8") as f:
        text = f.read()
    return [w for w in text.split() if w]

def save_vocab(vocab, rollno):
    fname = f"{rollno}_assignment2_wp_vocab_{len(vocab)}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in vocab:
            f.write(str(tok) + "\n")

def save_tokens(tokens, rollno):
    fname = f"{rollno}_assignment2_wp_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in tokens:
            f.write(str(tok) + "\n")

def save_detokenized(text, rollno):
    fname = f"{rollno}_assignment2_wp_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)



if __name__ == "__main__":

    from collections import defaultdict, Counter
    import heapq
    from collections import Counter, defaultdict
    import argparse
    from datetime import datetime
    from multiprocessing import Pool, cpu_count

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()

    rollno = "251110044"
    train_text = load_training_data(args.train)

    now = datetime.now()
    vocab = train_wordpiece(train_text, vocab_size=args.vocab_size)

    save_vocab(vocab, rollno)
    
    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = f.read()
    tokens = tokenize_text(sample_text, vocab)
    save_tokens(tokens, rollno)
    
    detok_text = detokenize(tokens)
    save_detokenized(detok_text, rollno)    
