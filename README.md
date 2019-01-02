# QA-ExtFeats

This repo contains the resources required to reproduce the results published in the SIGIR '17 short paper "On the Benefit of Incorporating External Features in a Neural Architecture for Answer Sentence Selection".

If you use the data or code from this repo, please cite the following paper:
```
@inproceedings{chen_benefit_2017,
 author = {Chen, Ruey-Cheng and Yulianti, Evi and Sanderson, Mark and Croft, W. Bruce},
 title = {On the Benefit of Incorporating External Features in a Neural Architecture for Answer Sentence Selection},
 booktitle = {Proceedings of {SIGIR} '17},
 year = {2017},
 pages = {1017--1020},
 publisher = {ACM}
} 
```

## Data ##

Feature vectors for two question answering benchmarks, TREC QA and WikiQA, are made available in the directory `data`.  The vectors are stored in the SVMLight format, and released in a compressed form (`.svm.gz`).

A complete list of 21 external features (and the respective feature IDs) are given as follows:

1. Length
2. ExactMatch
3. Overlap
4. OverlapSyn
5. LM
6. BM25
7. ESA
8. TAGME
9. Word2Vec
10. CPW
11. SPW
12. WPS
13. CWPS
14. CWR
15. LWPS
16. LWR
17. DaleChall
18. MatchedNGram[k=2,n=2]
19. MatchedNGram[k=2,n=3]
20. MatchedNGram[k=3,n=2]
21. MatchedNGram[k=3,n=3]

> The data is _NOT_ filtered in any way.  Some prior works remove questions with no positive answers, so additional care needs to be taken when making comparison with or incorporating this data into those approaches.
