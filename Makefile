.PHONY: check run bakeoff-schema

check:
	python3 -m shapefuse selfcheck
	python3 -m unittest discover -s tests -v

run:
	python3 -m shapefuse all --out results --device cpu --bench-level smoke

bakeoff-schema:
	python3 -m shapefuse bakeoff --out results/bakeoff-smoke --device cpu --level smoke --attention --skip-max-autotune
