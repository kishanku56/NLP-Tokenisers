from collections import defaultdict, Counter
import heapq
import argparse
from datetime import datetime
from multiprocessing import Pool, cpu_count
import math

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

def build_trie(vocab):
    root = TrieNode()
    for token in vocab:
        node = root
        for c in token:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_end = True
    return root

def compute_token_probs(vocab, corpus):
    vocab = set(vocab)
    token_counts = defaultdict(int)
    total_count = 0
    smoothing=1e-6
    max_token_len=10

    for word in corpus:
        L = len(word)
        for i in range(L):
            for j in range(i + 1, min(i + max_token_len + 1, L + 1)):
                token = word[i:j]
                if token in vocab:
                    token_counts[token] += 1
                    total_count += 1

    vocab_size = len(vocab)
    token_probs = { token: (token_counts[token] + smoothing) / (total_count + smoothing * vocab_size) for token in vocab}
    return token_probs

def k_best_viterbi_segmentation(word, token_probs, trie_root, k=2):

    n = len(word)    
    best_paths = [ [] for _ in range(n + 1) ]
    best_paths[0].append((0.0, []))     # Each position stores a min-heap of (negative log prob, path)  

    for i in range(n):
        if not best_paths[i]:
            continue

        node = trie_root
        for j in range(i, n):
            c = word[j]
            if c not in node.children:
                break
            node = node.children[c]

            if node.is_end:
                tok = word[i:j+1]
                tok_prob = token_probs.get(tok, 1e-12)
                log_tok_prob = math.log(tok_prob)

                for prev_log_prob, prev_path in best_paths[i]:
                    new_log_prob = prev_log_prob + log_tok_prob
                    new_path = prev_path + [tok]

                    heapq.heappush(best_paths[j+1], (new_log_prob, new_path))

                    # suppose k is 2, then length of best paths is 3. If we pop now, we remove the 3rd best path.
                    # So we are left with 2 best paths.

                    if len(best_paths[j+1]) > k:
                        heapq.heappop(best_paths[j+1])

    if not best_paths[n]:
        return [], []

    # Sort by descending probability, tie-breaker: lexicographic path
    sorted_paths = sorted(
        best_paths[n],
        key=lambda x: (x[0], x[1]),  # first log_prob, then path lexicographically
        reverse=True
    )
    segmentations = [path for log_prob, path in sorted_paths]
    probs = [math.exp(log_prob) for log_prob, path in sorted_paths]

    # Ensure exactly k outputs (fill with None if not enough)
    while len(segmentations) < k:
        segmentations.append(None)
        probs.append(1e-12)

    return segmentations[:k], probs[:k]




def process_token_chunk(args):
    chunk_tokens, vocab, corpus, token_probs, inverted_mapping, best_seg,second_best_seg,word_freqs = args
    results = {}
    
    for i, token_to_remove in enumerate(chunk_tokens, 1):
        affected_words = inverted_mapping.get(token_to_remove, [])
        ll_drop = 0.0

        removed_prob = token_probs.pop(token_to_remove, None)

        for word in affected_words:  # token_to_remove always in best_seg[word]
            if second_best_seg[word][0] is not None and token_to_remove not in second_best_seg[word][0]:
                new_prob = second_best_seg[word][1]
            else:
                new_prob = 1e-100

            ll_drop += word_freqs[word]*(math.log(new_prob) - math.log(best_seg[word][1]))

        if removed_prob is not None:
            token_probs[token_to_remove] = removed_prob

        # Store updated log-likelihood after removal
        results[token_to_remove] = ll_drop

    return results

def compute_token_removal_loglikelihood(vocab, corpus, token_probs, inverted_mapping, best_seg,second_best_seg,word_freqs,chunk_size):
    vocab = set(vocab)
    tokens = list(vocab)

    chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
    args_list = [(chunk, vocab, corpus, token_probs, inverted_mapping, best_seg,second_best_seg,word_freqs) for chunk in chunks]

    results = {}
    with Pool(cpu_count()) as pool:
        chunk_results = pool.map(process_token_chunk, args_list)
        for res in chunk_results:
            results.update(res)

    return results  


def compute_k_seg(token,token_probs,root_trie):
    segs = k_best_viterbi_segmentation(token, token_probs, root_trie, 2)
    return (token, segs)

def train_ulm_with_pruning(corpus, token_probs, target_vocab_size):

    word_freqs = Counter(corpus)
    unique_words = list(word_freqs.keys())

    while True:

        
        ''' 
        Finding all segmentations of a word is computationally intensive. Hence we use only the top 2 viterbi segmentations.
        see method process_token_chunk() to find how loss is computed using these 2 paths.
        Brief summary is given below:
        1. For each token in the vocab, store which words have this token in the best viterbi path. Token 
           removal will affect only these words. Let us call these words as affected words.
        2. For each affected word, check if the token is present in the second best viterbi path. 
        If it is not present, this is a good situation because token removal will not affect the second best path.
        loss=log(prob_second_best_path)-log(prob_best_path)
        3. If the token is present in the second best path as well, it means we lose both 1st best and 2nd best segmentations.
        This means high impact. loss = log(1e-100)-log(prob_best_path) ≈ -100 , very large negative value
        4. Step 2 and 3 is repeated for all words for a given token and the loss for each word is added up. This is loss for one token.
        5. We compute the loss for all tokens and finally remove the tokens with the least impact(pruning fraction is 0.2)
        '''

        num_processes = cpu_count()
        root_trie = build_trie(token_probs.keys())
        args = [(token, token_probs, root_trie) for token in unique_words]
        with Pool(num_processes) as pool:
            results = pool.starmap(compute_k_seg, args,chunksize=1000)

        # Convert back to dictionary
        k_segs = dict(results)


        best_segs={}
        alt_segs={}
        delete_words=[]
        for token in unique_words:
            if len(k_segs[token][0])==0:    #once we have pruned some tokens, some words might become unsegmentable in the next iteration.
                delete_words.append(token)
                continue

            best_segs[token]=(k_segs[token][0][0],k_segs[token][1][0])
            alt_segs[token]=(k_segs[token][0][1],k_segs[token][1][1])
        unique_words=set(unique_words).difference(delete_words)     # remove unsegmentable words
        inverted_mapping = defaultdict(list)
        best_seg_prob={}
        for word in unique_words:
            for tok in best_segs[word][0]:
                inverted_mapping[tok].append(word)  #stores which words have a particular token in their best path
                    
        results = compute_token_removal_loglikelihood(set(token_probs.keys()), corpus, token_probs,inverted_mapping,best_segs,alt_segs,word_freqs,max(100,int(len(token_probs)/10)))
        


        sorted_tokens = sorted(results.items(), key=lambda x: (-x[1], x[0]))
        tokens_to_remove = [(t,score) for t, score in sorted_tokens if len(t) > 1]
        next_size=int(max(target_vocab_size,0.8*len(token_probs)))
        number_of_tokens_to_be_deleted=len(token_probs)-next_size
        tokens_to_be_deleted=[]
        pruned_tokens=tokens_to_remove[:number_of_tokens_to_be_deleted]
        for token,prob in pruned_tokens:
            tokens_to_be_deleted.append(token)

        # Delete from token_probs
        for tok in tokens_to_be_deleted:
            if tok in token_probs:
                del token_probs[tok]
        
        # Renormalize probabilities
        total = sum(token_probs.values())
        for tok in token_probs:
            token_probs[tok] /= total
        

        if next_size==target_vocab_size:
            break
    
    return token_probs

def build_vocab(corpus, max_sub_len=10, min_freq=2):
    chars=set()        
    for word in corpus:
        for c in word:
            chars.add(c)

    required_tokens=20000-len(chars)
    word_freqs = Counter(corpus)
    sub_counter = Counter()
    for word, freq in word_freqs.items():
        L = len(word)
        for i in range(L):
            for j in range(i+2, min(i+max_sub_len+1, L+1)):
                sub_counter[word[i:j]] += freq

    top_subs = [sub for sub, freq in sub_counter.most_common() if freq >= min_freq]
    top_subs = top_subs[:required_tokens]

    seed_vocab = set(top_subs).union(chars)
    return list(seed_vocab)


def tokenize_text_parallel(text, token_probs, num_processes=None, chunksize=1000):
    root_trie = build_trie(token_probs.keys())
    words = text.split()

    args = [(word, token_probs, root_trie) for word in words]
    with Pool(cpu_count()) as pool:
        results = pool.starmap(tokenize_word, args, chunksize=1000)

    tokens = [t for sublist in results for t in sublist]
    return tokens



def tokenize_word(word, token_probs,root_trie):
    tokens = []
    seg, prob = k_best_viterbi_segmentation(word, token_probs, root_trie,1)
    if seg:
        tokens.append("▁"+seg[0][0])       # word boundary identification
        tokens.extend(seg[0][1:])
    else:
        tokens.append("<unk>")
    return tokens

def detokenize(tokens):
    text="".join(tokens).replace("▁"," ")
    if text and text[0]==" ":
        text=text[1:]
    return text

def load_training_data(train_path):
    with open(train_path, "r", encoding="utf-8") as f:
        text = f.read()
    return [w for w in text.split() if w]

def save_vocab(vocab, rollno):
    fname = f"{rollno}_assignment2_unigram_vocab_{len(vocab)}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in vocab:
            f.write(str(tok) + "\n")

def save_tokens(tokens, rollno):
    fname = f"{rollno}_assignment2_unigram_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in tokens:
            f.write(str(tok) + "\n")

def save_detokenized(text, rollno):
    fname = f"{rollno}_assignment2_unigram_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()

    rollno = "251110044"
    train_text = load_training_data(args.train)
    
    seed_vocab = build_vocab(train_text, max_sub_len=10, min_freq=2)
    #print(f"Initial seed vocab size: {len(seed_vocab)}")

    token_probs=compute_token_probs(seed_vocab,train_text)
    token_probs = train_ulm_with_pruning(train_text, token_probs,args.vocab_size)    
    save_vocab(list(sorted(token_probs.keys())), rollno)
    

    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = f.read()
    tokens = tokenize_text_parallel(sample_text, token_probs)
    save_tokens(tokens, rollno)
    
    detok_text = detokenize(tokens)
    save_detokenized(detok_text, rollno)
    
