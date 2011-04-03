import re
import networkx as nx
import numpy as np
from ConfigParser import ConfigParser
from collections import defaultdict, Counter
from operator import itemgetter
from scikits.learn.cluster import affinity_propagation

def create_graph():
    root = "/home/mxf/d/opinion_summarizer/OpinosisSummarizer-1.0/opinosis_sample/input/"
    filename = "bathroom_bestwestern_hotel_sfo.txt.data.parsed"
    #filename = "food_holiday_inn_london.txt.data.parsed"
    review_file = root + filename
    #review_file = "tmp_reviews"
    edges = []
    nodes_pri = defaultdict(list)
    with open(review_file, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            words = line.split()
            words2 = words[1:][:]
            words1 = words[:-1]
            bigram = zip(words1, words2)
            edges.extend(bigram)
            for j, word in enumerate(words):
                nodes_pri[word].append((i,j))
    edges_cnt = Counter(edges)
    return edges_cnt, nodes_pri

def valid_start_node(node, nodes_pri):
    """
    Determine if node is a valid start node
    """
    start_tag = set(["JJ", "RB", "PRP$", "VBG", "NN", "DT"])
    start_word = set(["its", "the", "when", "a", "an", "this", 
                      "the", "they", "it", "i", "we", "our",
                      "if", "for"])
    pri = nodes_pri[node]
    position = [e[1] for e in pri]
    median = np.median(position)
    START = int(cp.get("section", "start"))
    if median <= START:
        w, t = node.split("/")
        if w in start_word or t in start_tag:
            return True
    return False

def intersect(pri_so_far, pri2):
    """
    Intersect two list of path redundancy: (sid, pid).
    """
    GAP = int(cp.get("section", "gap"))
    pri_new = []
    for pri in pri_so_far:
        last_sid, last_pid = pri[-1]
        for sid, pid in pri2:
            if sid == last_sid and pid - last_pid > 0 and pid - last_pid <= GAP:
                pri = pri[:]
                pri.append((sid, pid))
                pri_new.append(pri)
    return pri_new

def valid_end_node(graph, node):
    if "/." in node  or "/," in node:
        return True
    elif len(graph[node]) <= 0:
        return True
    else:
        return False

def valid_candidate(sentence):
    #return True
    sent = " ".join(sentence)
    last = sentence[-1]
    w, t = last.split("/")
    if t in set(["TO", "VBZ", "IN", "CC", "WDT", "PRP", "DT" ","]):
        return False
    if re.match(".*(/JJ)*.*(/NN)+.*(/VB)+.*(/JJ)+.*", sent):
        return True
    elif re.match(".*(/RB)*.*(/JJ)+.*(/NN)+.*", sent) and not re.match(".*(/DT).*", sent):
        return True
    elif re.match(".*(/PRP|/DT)+.*(/VB)+.*(/RB|/JJ)+.*(/NN)+.*", sent):
        return True
    elif re.match(".*(/JJ)+.*(/TO)+.*(/VB).*", sent):
        return True
    elif re.match(".*(/RB)+.*(/IN)+.*(/NN)+.*", sent):
        return True
    else:
        return False

def path_score(redundancy, sen_len):
    """
    log weghted redundancy score function
    """
    if sen_len == 2:
        return redundancy
    else:
        return np.log2(sen_len) * redundancy

def collapsible(node):
    #return False
    if re.match(".*(/VB[A-Z]|/IN)", node):
        return True
    else:
        return False

def average_path_score(cc):
    return np.mean(cc.values())

def intersection_sim(can1, can2):
    set1 = set(can1.split())
    set2 = set(can2.split())

    return float(len(set1.intersection(set2)))/len(set1.union(set2))

def remove_duplicates(cc, sim_func=intersection_sim):
    li = cc.keys()
    sim_matrix = np.zeros((len(li), len(li)))
    for i, e1 in enumerate(li):
        for j, e2 in enumerate(li):
            sim_matrix[i,j] = sim_func(e1, e2)

    centers, _ = affinity_propagation(sim_matrix)

    for i, e in enumerate(li):
        if i not in centers:
            del cc[e]

def stitch(canchor, cc):
    if len(cc) == 1:
        return cc.keys()[0]
    #remove_duplicates(cc)
    return " xx ".join(cc.keys())
    sents = cc.keys()
    anchor_str = " ".join(canchor)
    anchor_len = len(anchor_str)
    sents = [e[anchor_len:] for e in sents]
    sents = [e for e in sents if e.strip() != "./." and e.strip() != ",/,"]
    s = anchor_str + " xx " + " AND ".join(sents)
    return s + " ."

def traverse(graph, nodes_pri, node, sentence, pri_so_far, score, clist, collapsed):
    """
    traverse a path
    """
    if len(sentence) > 20:
        return 
    redundancy = len(pri_so_far)
    REDUNDANCY_THRESHOLD = int(cp.get("section", "redundancy"))
    if redundancy >= REDUNDANCY_THRESHOLD or valid_end_node(graph, node):
        if valid_end_node(graph, node):
            if valid_candidate(sentence):
                final_score = score/float(len(sentence))
                clist[" ".join(sentence)] = final_score
                #print sentence, pri_so_far
            return

        # Traversing the neighbors
        for neighbor in graph[node]:
            pri_new = intersect(pri_so_far, nodes_pri[neighbor])
            #print pri_so_far
            redundancy = len(pri_so_far)
            new_sentence = sentence[:]
            new_sentence.append(neighbor)
            new_score = score + path_score(redundancy, len(new_sentence))
            
            if collapsible(neighbor) and not collapsed:
                #print new_sentence, pri_new
                canchor = new_sentence
                cc = defaultdict(int)
                for vx in graph[neighbor]:
                    pri_vx = intersect(pri_new, nodes_pri[vx])
                    vx_sentence = new_sentence[:]
                    vx_sentence.append(vx)
                    traverse(graph, nodes_pri, vx, vx_sentence, 
                             pri_vx, new_score, cc, True)
                cc_path_score = average_path_score(cc)
                final_score = new_score + cc_path_score
                if cc:
                    stitched_sent = stitch(canchor, cc)
                    clist[stitched_sent] = final_score
            else:
                traverse(graph, nodes_pri, neighbor, new_sentence,
                         pri_new, new_score, clist, False)

def summarize(graph, nodes_pri):
    """
    Summerizing a graph
    """
    nodes_size = len(nodes_pri)
    candidates = defaultdict(int)
    for node in nodes_pri:
        if valid_start_node(node, nodes_pri):
            path_len = 1
            score = 0
            clist = defaultdict(int)
            sentence = [node]
            pri = nodes_pri[node]
            pri_so_far = [[e] for e in pri] 
            traverse(graph, nodes_pri, node, sentence, pri_so_far, score, clist, False)
            candidates.update(clist)

    return candidates

cp = None
if __name__ == '__main__':
    edges_cnt, nodes_pri = create_graph()

    with open('review_edges', 'w') as f:
        for bigram in edges_cnt:
            f.write(" ".join([bigram[0], bigram[1]]) + " " + str(edges_cnt[bigram]))
            f.write("\n")

    G = nx.read_edgelist('review_edges', create_using=nx.DiGraph(),data=(('count',int),))
    cp = ConfigParser()
    cp.read("opinosis.properties")
    candidates = summarize(G, nodes_pri)

    #remove_duplicates(candidates)

    li = candidates.items()
    li.sort(key=itemgetter(1), reverse=True)
    for e in li:
        print e[0], e[1]
