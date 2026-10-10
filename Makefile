.PHONY: check run

check:
	python3 -m shapefuse selfcheck
	python3 -m unittest discover -s tests -v

run:
	python3 -m shapefuse all --out results --device cpu --bench-level smoke
