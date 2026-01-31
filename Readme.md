# llmdata

Lightweight, config-driven tooling to validate and prepare text datasets for LLM pretraining.

This repository currently focuses on **Step 1: small-sample download and verification** for large public datasets, to ensure:
- correct dataset configs / splits
- successful downloads
- consistent, parseable JSONL outputs

---

## Supported datasets (Step 1)

- **Sangraha (AI4Bharat)**  
  - Config: `verified`
  - Language splits (e.g. `hin`, `eng`)
- **IndicCorpV2 (AI4Bharat)**  
  - Language-script splits (e.g. `hin_Deva`, `ben_Beng`)
- **Dolma (AllenAI)**  
  - Supported via shard download (optional; disabled by default)

---

## Repository structure

.
├── configs/
│ └── sample.toml # dataset configs + sample sizes
├── src/
│ ├── sample_download.py # downloads small samples
│ └── sample_verify.py # verifies downloaded samples
├── out/ # generated data (gitignored)
├── .gitignore
└── README.md



---

## Step 1: Sample download

Edit `configs/sample.toml` to control:
- number of records per dataset
- dataset splits / languages
- whether to include Dolma shards

Run:
```bash
python src/sample_download.py
