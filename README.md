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

Feature vectors for two question answering benchmarks, TREC QA and WikiQA, are made available in the directory `data`.
The vectors are stored in the SVMLight format, and released in a compressed form (`.svm.gz`).

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

## Feature Extraction ##

Feature extraction scripts (**in python 3**) are also available in the `data` directory. 
Follow these steps to rerun the extraction pipeline:

- Extract frequency counts from the GOV2 collection, and store the counts in compressed format
  in the file `freqstats.gz`. One can simply index the collection using Indri and then use its
  utility `dump_index` to do the work:

      dumpindex /path/to/index v | gzip > freqstats.gz
   
- Build an Indri index over English Wikipedia pages. This can be done by first downloading an [enwiki data
  dump](https://dumps.wikimedia.org/enwiki/) and then converting wikipages into TRECTEXT format,
  using DOCNO of the form `ENWIKI_XXXXXXXX` (where `XXXXXXXX` indicates the wikipage number).
  
  The enwiki dump used in our experiment was produced in October 2015.
  
- Download and uncompress the pretrained word2vec model [GoogleNews-vectors-negative300.bin](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
  
- Sign up for the [TAGME service](https://services.d4science.org/web/tagme/tagme-help) to obtain a TAGME token.
  
- Modify the _Configurations_ section of `Makefile` to reflect the local changes.
  Then run the task specific targets:

      make trecqa
      make wikiqa

## Experiments ##

To reproduce our results:

- Clone the [deep-qa][] package (commit `ca7b5079160908c3e2356dd735c90666fac12a21`).
  Drop the python scripts (**in python 2**) under `cnn` into the `deep-qa` repo.

- In the [deep-qa] repo, follow through the data preparation steps to produce embeddings and model input.
  A model directory (e.g. `TRAIN-ALL` for TREC QA data) will be made available upon completion.

- Generate feature files (`*.svm.gz`) following the previous instructions.
  Then use `python2 svmlight_to_npy.py` to convert features for train/dev/test sets into numpy format,
  and then drop those files back into the model directory. For example, in the TREC QA experiment 
  three files are needed `{train-all,dev,test}.overlap_feats.npy`.

- Experiments can now be carried out from the `deep-qa` repo using `python2 run_nnet.py`.
  (Most of the hyperparameters can be changed through the command-line interface.)

  > The script will also look for files `*.aids.npy` (answer IDs) in the model directory to generate
  > proper TREC run output. One may need to modify `parse.py` to provide such information.

[deep-qa]: https://github.com/aseveryn/deep-qa
