# GNU Makefile 


#------------------------------------------------------------------------------
# LOCAL VARS
#------------------------------------------------------------------------------

# CONTAINER
DOCKER := podman
MPNET := llms/all-mpnet-base-v2

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
# ALL TARGETS
#------------------------------------------------------------------------------
.PHONY: all directories clean

TARGET1a := derivatives/sentences_gpt-4o/list_of_simplified.txt
TARGET1b := derivatives/sentences_gpt-4o/list_of_detailed.txt
TARGET2 := derivatives/sentences_gpt-4o/list_of_df.txt
TARGET3 := derivatives/sentences_gpt-4o_embed-mpnet/list_of_files.txt
TARGET4a := derivatives/sentences_gpt-4o_embed-mpnet/list_of_cosine_files.txt
TARGET4b := derivatives/sentences_gpt-4o_embed-mpnet/list_of_tsne_files.txt
TARGET5 := derivatives/sentences_gpt-4o_classfier/gpt-4o_sentences.parquet
TARGET6a := derivatives/sentences_gpt-4o_classfier/clf-rforests_simplified.joblib
TARGET6b := derivatives/sentences_gpt-4o_classfier/clf-rforests_detailed.joblib
TARGET6c := derivatives/sentences_gpt-4o_classfier/clf-rforests_depth.joblib
TARGET6d := derivatives/sentences_gpt-4o_classfier/clf-rforests_all.joblib
TARGET7 := derivatives/sentences_gpt-4o_classfier/clf-outlier.joblib


all: \
	directories \
	$(TARGET1a) \
	$(TARGET1b) \
	$(TARGET2) \
	$(TARGET3) \
	$(TARGET4b) \
	$(TARGET5) \
	$(TARGET6a) \
	$(TARGET6b) \
	$(TARGET6c) \
	$(TARGET6d) \
	# $(TARGET7)


directories:
	mkdir -p derivatives/sentences_gpt-4o
	mkdir -p derivatives/sentences_gpt-4o_embed-mpnet
	mkdir -p derivatives/sentences_gpt-4o_classfier

clean:
	rm -fr derivatives/sentences_gpt-4o*

#------------------------------------------------------------------------------
# TARGETS
#------------------------------------------------------------------------------

$(TARGET1a): \
	code/sentences_gpt-4o/get_sentences_gpt.py \
	code/sentences_gpt-4o/prompts/prompt_system_simplified.txt \
	stimulus/AVALANCHES.txt \
	code/utilities_py/utility_openai.py \
	code/utilities_py/utility_functions.py \
	| code/utilities_py
	$(run-py) $(wordlist 1, 3, $^) $| $@ simplified

$(TARGET1b): \
	code/sentences_gpt-4o/get_sentences_gpt.py \
	code/sentences_gpt-4o/prompts/prompt_system_detailed.txt \
	stimulus/AVALANCHES.txt \
	code/utilities_py/utility_openai.py \
	code/utilities_py/utility_functions.py \
	| code/utilities_py
	$(run-py) $(wordlist 1, 3, $^) $| $@ detailed

$(TARGET2): \
	code/sentences_gpt-4o/preproc_sentences_gpt.py \
	$(TARGET1a) \
	$(TARGET1b) \
	code/utilities_py/utility_functions.py \
	| code/utilities_py
	$(run-py) $(wordlist 1, 3, $^) $| $@

$(TARGET3): \
	code/sentences_gpt-4o/get_embedding_sbert.py \
	$(TARGET2) \
	code/utilities_py/utility_sbert.py \
	| $(MPNET) \
	code/utilities_py
	$(run-pytorch) $(wordlist 1, 2, $^) $| $@

$(TARGET4a): \
	code/sentences_gpt-4o/plot_mpnet_embedding_cosine.py \
	$(TARGET3) 
	$(run-pytorch) $^ $@

$(TARGET4b): \
	code/sentences_gpt-4o/plot_mpnet_embedding_tsne.py \
	$(TARGET3) 
	$(run-pytorch) $^ $@

$(TARGET5): \
	code/sentences_gpt-4o/concat_sentences_gpt.py \
	$(TARGET3) 
	$(run-pytorch) $^ $@

$(TARGET6a): \
	code/sentences_gpt-4o/train_test_classfier.py \
	$(TARGET5) 
	$(run-pytorch) $^ $@ simplified

$(TARGET6b): \
	code/sentences_gpt-4o/train_test_classfier.py \
	$(TARGET5) 
	$(run-pytorch) $^ $@ detailed

$(TARGET6c): \
	code/sentences_gpt-4o/train_test_classfier_depth.py \
	$(TARGET5) 
	$(run-pytorch) $^ $@

$(TARGET6d): \
	code/sentences_gpt-4o/train_test_classfier_all.py \
	$(TARGET5) 
	$(run-pytorch) $^ $@

$(TARGET7): \
	code/sentences_gpt-4o/outlier_detection.py \
	$(TARGET5) 
	$(run-pytorch) $^ $@