# Synthetic Demographics

This repository contains tools and data for generating synthetic demographic data.

## Installation

This repository contains large files managed with Git LFS. To clone and use this repository:

1. Install Git LFS:
   - MacOS: `brew install git-lfs`
   - Ubuntu/Debian: `sudo apt-get install git-lfs`
   - Windows: `choco install git-lfs`

2. Clone the repository:
   ```bash
   git clone https://github.com/C0deMunk33/synthetic_demographics.git
   cd synthetic_demographics
   git lfs pull
   ```

3. Install with poetry
    ```bash
    poetry install
    poetry run postinstall
    ```

Note: The output_database.zip file is approximately 89MB. Make sure you have sufficient storage space and a stable internet connection when cloning.

## Run
```bash
    poetry run python synthetic_demographics/demographic_generator.py
```

This will create 100 synthetic demographics, and output `synthetic_demographics.jsonl`

adjust `batch_size` and `batch_count` as desired

In python:
```python
database_name = "output_database.db"
demographic_generator = DemographicGenerator(db_path=database_name)
results = demographic_generator.generate_demographic_batch(samples=5)
```
demographic_generator.generate_demographic_batch(samples=batch_size) 

## Large Files

This repository uses Git Large File Storage (Git LFS) to handle large files. The following files are tracked using Git LFS:
- `synthetic_demographics/output_database.zip` (89MB)

If you're having trouble accessing these files, make sure you have Git LFS installed and have run `git lfs pull` after cloning the repository.

## License

MIT
