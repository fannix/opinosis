import networkx as nx
import numpy as np
from ConfigParser import ConfigParser
from collections import defaultdict, Counter
from operator import itemgetter

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
    pri = nodes_pri[node]
    position = [e[1] for e in pri]
    median = np.median(position)
    START = int(cp.get("section", "start"))
    if median <= START:
        return True
    else:
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
                pri.append((sid, pid))
                pri_new.append(pri[:])
    return pri_new

def valid_end_node(node):
    if node == "./." or node == ",/,"\
       or node == "but/CC" or node == "and/CC" or node == "yet/CC":
        return True
    else:
        return False

def valid_sentence(sentence):
    return True

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
    if node == "is/VBZ" or node == "are/VBP"\
       or node == "was/VBD" or node == "were/VBD":
        return True
    else:
        return False

def average_path_score(cc):
    return np.mean(cc.values())

def stitch(canchor, cc):
    s = " ".join(canchor) + " " + " and " + " ".join(cc.keys())
    return s

def traverse(graph, nodes_pri, node, sentence, pri_so_far, score, clist):
    """
    traverse a path
    """
    if len(sentence) > 20:
        return 
    redundancy = len(pri_so_far)
    REDUNDANCY_THRESHOLD = int(cp.get("section", "redundancy"))
    if redundancy >= REDUNDANCY_THRESHOLD:
        if valid_end_node(node):
            if valid_sentence(sentence):
                final_score = score/float(len(sentence))
                clist[" ".join(sentence)] = final_score
                #print sentence, pri_so_far

        # Traversing the neighbors
        for neighbor in graph[node]:
            pri_new = intersect(pri_so_far, nodes_pri[neighbor])
            #print pri_so_far
            redundancy = len(pri_so_far)
            new_sentence = sentence[:]
            new_sentence.append(neighbor)
            new_score = score + path_score(redundancy, len(new_sentence))
            
            if collapsible(neighbor):
                canchor = new_sentence
                cc = defaultdict(int)
                for vx in graph[neighbor]:
                    li = [vx]
                    pri_vx = intersect(pri_new, nodes_pri[vx])
                    traverse(graph, nodes_pri, vx, li, pri_vx, 0, cc)
                cc_path_score = average_path_score(cc)
                final_score = new_score + cc_path_score
                if cc:
                    stitched_sent = stitch(canchor, cc)
                    clist[stitched_sent] = final_score
                else:
                    traverse(graph, nodes_pri, neighbor, new_sentence,
                             pri_new, new_score, clist)
            else:
                traverse(graph, nodes_pri, neighbor, new_sentence,
                         pri_new, new_score, clist)

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
            traverse(graph, nodes_pri, node, sentence, pri_so_far, score, clist)
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

    li = candidates.items()
    li.sort(key=itemgetter(1), reverse=True)
    for e in li:
        print e
