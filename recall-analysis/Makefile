# GNU Makefile 


#------------------------------------------------------------------------------
# LOCAL VARS
#------------------------------------------------------------------------------

# CONTAINER
DOCKER := podman

#------------------------------------------------------------------------------
# DOCKER UTILS
#------------------------------------------------------------------------------
.PHONY: test-docker-py test-docker-tf test-docker-pytorch

test-docker-py:
	$(DOCKER) run --rm --user=root \
	-it -v `pwd`:/home/dockeruser python:avalanche-env /bin/bash

test-docker-pytorch:
	$(DOCKER) run --rm -it -v "${PWD}":/home/jovyan/work \
	pytorch:avalanche-env /bin/bash

test-docker-tf:
	$(DOCKER) run --rm -it -v "${PWD}":/home/jovyan/work \
	quay.io/jupyter/tensorflow-notebook:aarch64-python-3.11.9 /bin/bash


# short-cut to run the docker container
run-py := $(DOCKER) run --rm --user=root \
	-it -v `pwd`:/home/dockeruser python:avalanche-env python

run-pytorch := $(DOCKER) run -it --rm -v "${PWD}":/home/jovyan \
	pytorch:avalanche-env python

run-tf := $(DOCKER) run -it --rm -v "${PWD}":/home/jovyan/work \
	quay.io/jupyter/tensorflow-notebook:aarch64-python-3.11.9 python

#------------------------------------------------------------------------------
# ALL
#------------------------------------------------------------------------------
.PHONY: all directories clean

NLTK := nltk
SPACY := spacy/en_core_web_sm
MPNET := llms/all-mpnet-base-v2
CLFa := derivatives/sentences_gpt-4o_classfier/clf-rforests_simplified.joblib
CLFb := derivatives/sentences_gpt-4o_classfier/clf-rforests_detailed.joblib
CLFc := derivatives/sentences_gpt-4o_classfier/clf-rforests_depth.joblib
CLFd := derivatives/sentences_gpt-4o_classfier/clf-rforests_all.joblib
CLF_LOF := derivatives/sentences_gpt-4o_classfier/clf-lof.joblib

DATADIR := data-recall
TARGET1 := derivatives/recall-analysis/tokenized/list_of_files.txt
TARGET2 := derivatives/recall-analysis/sbert/list_of_files.txt
TARGET3a := derivatives/recall-analysis/clf/list_of_files_simplified.txt
TARGET3b := derivatives/recall-analysis/clf/list_of_files_detailed.txt
TARGET3c := derivatives/recall-analysis/clf/list_of_files_depth.txt
TARGET3d := derivatives/recall-analysis/clf/list_of_files_all.txt
TARGET4 := derivatives/recall-analysis/clf/list_of_files_outlier.txt

TEXT := stimulus/AVALANCHES.csv
TARGET_0 := derivatives/recall-analysis/sbert/text.parquet
TARGET_1a := derivatives/recall-analysis/clf/text_simplified.joblib
TARGET_1b := derivatives/recall-analysis/clf/text_detailed.joblib
TARGET_1d := derivatives/recall-analysis/clf/text_all.joblib

all: \
	directories \
	$(TARGET1) \
	$(TARGET2) \
	$(TARGET3a) \
	$(TARGET3b) \
	$(TARGET3c) \
	$(TARGET3d) \
	# $(TARGET_0) \
	# $(TARGET_1a) \
	# $(TARGET_1b) \
	# $(TARGET_1d) \
	# $(TARGET4)


directories:
	mkdir -p derivatives/recall-analysis/tokenized
	mkdir -p derivatives/recall-analysis/sbert
	mkdir -p derivatives/recall-analysis/clf

clean:
	rm -fr derivatives/recall-analysis

#------------------------------------------------------------------------------
# TARGETS
#------------------------------------------------------------------------------

$(TARGET1): \
	code/recall-analysis/tokenize_recall.py \
	$(DATADIR) \
	$(SPACY) \
	code/utilities_py/utility_functions.py \
	| code/utilities_py
	$(run-py) $(wordlist 1, 3, $^) $| $@

$(TARGET2): \
	code/recall-analysis/sbert_recall.py \
	$(TARGET1) \
	$(MPNET) \
	code/utilities_py/utility_sbert.py \
	| code/utilities_py
	$(run-pytorch) $(wordlist 1, 3, $^) $| $@

$(TARGET3a): \
	code/recall-analysis/run_clf.py \
	$(TARGET2) \
	$(CLFa)
	$(run-pytorch) $^ $@ simplified

$(TARGET3b): \
	code/recall-analysis/run_clf.py \
	$(TARGET2) \
	$(CLFb)
	$(run-pytorch) $^ $@ detailed

$(TARGET3c): \
	code/recall-analysis/run_clf.py \
	$(TARGET2) \
	$(CLFc)
	$(run-pytorch) $^ $@ depth

$(TARGET3d): \
	code/recall-analysis/run_clf.py \
	$(TARGET2) \
	$(CLFd)
	$(run-pytorch) $^ $@ all

$(TARGET4): \
	code/recall-analysis/run_clf_outlier.py \
	$(TARGET2) \
	$(CLF_LOF)
	$(run-pytorch) $^ $@

$(TARGET_0): \
	code/recall-analysis/sbert_text.py \
	$(TEXT) \
	$(MPNET) \
	code/utilities_py/utility_sbert.py \
	| code/utilities_py
	$(run-pytorch) $(wordlist 1, 3, $^) $| $@

$(TARGET_1a): \
	code/recall-analysis/run_clf_text.py \
	$(TARGET_0) \
	$(CLFa)
	$(run-pytorch) $^ $@ simplified

$(TARGET_1b): \
	code/recall-analysis/run_clf_text.py \
	$(TARGET_0) \
	$(CLFb)
	$(run-pytorch) $^ $@ detailed

$(TARGET_1d): \
	code/recall-analysis/run_clf_text.py \
	$(TARGET_0) \
	$(CLFd)
	$(run-pytorch) $^ $@ all