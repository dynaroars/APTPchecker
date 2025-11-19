NEURALSAT_DIR = /home/roars/neuralsat_proof/src

all: neuralsat

neuralsat: neuralsat_input neuralsat_hidden

neuralsat_input:
	python3 runner.py --verifier neuralsat --split_type input --verifier_dir $(NEURALSAT_DIR)

neuralsat_hidden:
	python3 runner.py --verifier neuralsat --split_type hidden --verifier_dir $(NEURALSAT_DIR)