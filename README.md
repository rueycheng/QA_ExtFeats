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

Feature vectors for two question answering benchmarks, TREC QA and WikiQA, are made available in the directory `data`.  The vectors are stored in the SVMLight format, and released in a compressed form (`.gz`).

The data is _NOT_ filtered in any way.  Some prior works remove questions with no positive answers, so additional care needs to be taken when making comparison with or incorporating this data into those approaches.

A complete list of 22 external features (and the respective feature IDs) are given as follows:

1. Length
2. Location
3. ExactMatch
4. Overlap
5. OverlapSyn
6. LM
7. BM25
8. ESA
9. TAGME
10. Word2Vec
11. CPW
12. SPW
13. WPS
14. CWPS
15. CWR
16. LWPS
17. LWR
18. DaleChall
19. MatchedNGram[k=2,n=2]
20. MatchedNGram[k=2,n=3]
21. MatchedNGram[k=3,n=2]
22. MatchedNGram[k=3,n=3]

## Note ##

Code for feature extraction is to be made available soon.
