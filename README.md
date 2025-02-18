# synthetic_demographics

## Install

```bash
poetry install
poetry run postinstall
poetry run python synthetic_demographics/demographic_generator.py
```

This will create 100 synthetic demographics, and output `synthetic_demographics.jsonl`

adjust `batch_size` and `batch_count` as desired

In python:
```
database_name = "output_database.db"
demographic_generator = DemographicGenerator(db_path=database_name)
results = demographic_generator.generate_demographic_batch(samples=5)
```
demographic_generator.generate_demographic_batch(samples=batch_size) 