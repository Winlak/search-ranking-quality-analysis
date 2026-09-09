PYTHON ?= python3

.PHONY: install test reproduce check

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

reproduce:
	$(PYTHON) -m src.pipeline

check: test reproduce

