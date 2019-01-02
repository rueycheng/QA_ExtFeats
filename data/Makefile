PYTHON = python3


# Configurations
#
ESA_INDEX_PATH = ~/work/indexes/indri_index.enwiki
WORD2VEC_MODEL = ~/work/embeddings/GoogleNews-vectors-negative300.bin
TAGME_TOKEN = xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx


# Main targets
#
.DELETE_ON_ERROR:

.PHONY: help wikiqa trecqa clean

help:
	@echo "Available targets: wikiqa trecqa"

wikiqa: WikiQA-train.tsv WikiQA-dev.tsv WikiQA-test.tsv
	$(PYTHON) extract_features.py \
	    -m repo_wikiqa \
	    --bm25-avg-dl 10 \
	    --esa-index-path $(ESA_INDEX_PATH) \
	    --esa-k 100 \
	    --esa-threads 12 \
	    --word2vec-model $(WORD2VEC_MODEL) \
	    --tagme-token $(TAGME_TOKEN) \
	    --tagme-threads 40 \
	    -- \
	    wikiqa $+
	cat repo_wikiqa/output.train | grep -v '^#' | gzip > wikiqa-train.svm.gz
	cat repo_wikiqa/output.dev | grep -v '^#' | gzip > wikiqa-dev.svm.gz
	cat repo_wikiqa/output.test | grep -v '^#' | gzip > wikiqa-test.svm.gz

trecqa: trecqa-train-all.xml.gz trecqa-train.xml trecqa-dev.xml trecqa-test.xml
	$(PYTHON) extract_features.py \
	    -m repo_trecqa \
	    --bm25-avg-dl 10 \
	    --esa-index-path $(ESA_INDEX_PATH) \
	    --esa-k 100 \
	    --esa-threads 12 \
	    --word2vec-model $(WORD2VEC_MODEL) \
	    --tagme-token $(TAGME_TOKEN) \
	    --tagme-threads 40 \
	    -- \
	    trecqa $+
	cat repo_trecqa/output.train_all | grep -v '^#' | gzip > trecqa-train_all.svm.gz
	cat repo_trecqa/output.train | grep -v '^#' | gzip > trecqa-train.svm.gz
	cat repo_trecqa/output.dev | grep -v '^#' | gzip > trecqa-dev.svm.gz
	cat repo_trecqa/output.test | grep -v '^#' | gzip > trecqa-test.svm.gz

clean:
	rm -f wikiqa-train.svm.gz wikiqa-dev.svm.gz wikiqa-test.svm.gz
	rm -f trecqa-train_all.svm.gz trecqa-train.svm.gz trecqa-dev.svm.gz trecqa-test.svm.gz


# Sources
#
.INTERMEDIATE: WikiQACorpus.zip jacana-qa-naacl2013-data-results.tar.bz2

WikiQACorpus.zip:
	wget https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip -O $@

jacana-qa-naacl2013-data-results.tar.bz2:
	wget http://cs.jhu.edu/~xuchen/packages/jacana-qa-naacl2013-data-results.tar.bz2 -O $@

.INTERMEDIATE: WikiQA-train.tsv WikiQA-dev.tsv WikiQA-test.tsv

WikiQA-train.tsv WikiQA-dev.tsv WikiQA-test.tsv: WikiQACorpus.zip
	unzip -p $< WikiQACorpus/$@ > $@

.INTERMEDIATE: trecqa-train.xml trecqa-dev.xml trecqa-test.xml trecqa-train-all.xml.gz

trecqa-train.xml: jacana-qa-naacl2013-data-results.tar.bz2
	tar -jxOvf $< jacana-qa-naacl2013-data-results/train-less-than-40.manual-edit.xml > $@
trecqa-dev.xml: jacana-qa-naacl2013-data-results.tar.bz2
	tar -jxOvf $< jacana-qa-naacl2013-data-results/dev-less-than-40.manual-edit.xml > $@
trecqa-test.xml: jacana-qa-naacl2013-data-results.tar.bz2
	tar -jxOvf $< jacana-qa-naacl2013-data-results/test-less-than-40.manual-edit.xml > $@
trecqa-train-all.xml.gz: jacana-qa-naacl2013-data-results.tar.bz2
	tar -jxOvf $< jacana-qa-naacl2013-data-results/train2393.cleanup.xml.gz > $@
