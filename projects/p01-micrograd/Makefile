# Uses '>' as the recipe prefix so this file needs no literal tabs.
.RECIPEPREFIX = >
PY ?= python

.PHONY: help test api gradcheck parity accept clean

help:
> @echo "make test       - run every suite"
> @echo "make api        - semantics only (no torch needed)"
> @echo "make gradcheck  - finite-difference check only (no torch needed)"
> @echo "make parity     - torch.autograd comparison"
> @echo "make accept     - the full acceptance run (this is the one that counts)"

test:
> $(PY) -m pytest -q tests

api:
> $(PY) -m pytest -q tests/test_api.py

gradcheck:
> $(PY) -m pytest -q tests/test_gradcheck.py

parity:
> $(PY) -m pytest -q tests/test_torch_parity.py

accept:
> $(PY) eval/acceptance.py

clean:
> rm -rf .pytest_cache tests/__pycache__ src/micrograd/__pycache__ eval/__pycache__
