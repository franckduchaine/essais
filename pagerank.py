import os
import random
import re
import sys
import numpy as np

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    
    distribution = dict()
    
    d=damping_factor
    N=len(corpus)
    number_of_links=len(corpus[page])
    
    
    for the_page in corpus:
        if the_page in corpus[page]:
            distribution[the_page]=(1-d)/N+d*(1/number_of_links)
        else :
            distribution[the_page]=(1-d)/N
    
    
    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    assert(n!=0)
    N=len(corpus)
    
    page_rank_sample=dict()
    
    for the_page in corpus:
        page_rank_sample[the_page]=0
        
    current_page=random.choice(list(corpus.keys()))
    
    

    
    for i in range(n):
        page_rank_sample[current_page]=page_rank_sample[current_page]+1
        transition=transition_model(corpus, current_page, damping_factor)
        tirage=random.random()
        seuil=0
        
        for the_page in transition:
            seuil=seuil+transition[the_page]
            if tirage<=seuil :
                current_page=the_page
                break
    
    for the_page in page_rank_sample:
        page_rank_sample[the_page]=page_rank_sample[the_page]/n
    

    return page_rank_sample
    


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    
    page_rank_markov
    """
    

    N=len(corpus)
    epsilon=0.001
    
    delta=float('infinity')
    
    t=matrice_transition(corpus, damping_factor)
    

    current_page_rank=(1/N)*np.ones((N))
    
    
    
    while delta>epsilon:

        new_page_rank=np.dot(t,current_page_rank)
        delta=max(np.max(new_page_rank-current_page_rank),np.max(current_page_rank-new_page_rank))
        current_page_rank=new_page_rank
        
    page_rank_markov=dict()
    
    indice=0
    for the_page in corpus:
        page_rank_markov[the_page]=current_page_rank[indice]
        indice=indice+1
        
        
    return page_rank_markov
    
    


def reverse_corpus(corpus):
    r_corpus=dict()
    for the_page in corpus:
        r_corpus[the_page]=set()
    for key, values in corpus.items():
        for page in values:
            r_corpus[page].add(key)
    
    return r_corpus
    
def matrice_transition(corpus, damping_factor):

    N=len(corpus)
    r_corpus=reverse_corpus(corpus)
    
    t=((1-damping_factor)/N)*np.ones((N,N))
    
    i=0
    p=0
    
    for  state_p in corpus:
        for state_i in corpus:
            if state_i in r_corpus[state_p]:
                t[p][i]=t[p][i]+damping_factor/len(list(corpus[state_i]))
            i=i+1
        i=0
        p=p+1
    return t
            
    
    

if __name__ == "__main__":
    main()
