.PHONY: setup data profile features-name features-record features test all clean

setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

data:
	python scripts/01_parse.py

profile:
	python scripts/02_profile.py

features-name:
	python scripts/03a_name_features.py

features-record:
	python scripts/03b_record_features.py

features: features-name features-record

test:
	pytest -q

all: data profile test

clean:
	rm -f data/interim/*.parquet data/processed/*.parquet
