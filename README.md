# 🎌 Anime Data Pipeline

A full end-to-end data science pipeline built with Python — raw messy anime data → clean, ML-ready CSVs.

## 📦 Dataset
- **100 anime** | **207 season records** | intentionally messy raw data
- Issues handled: duplicates, missing values, mixed types, string nulls

## ⚙️ Pipeline Steps
1. **Raw Data** → Parse tuples into DataFrames
2. **Cleaning** → Dedup, fix types, replace `None`/`N/A` with `NaN`
3. **Imputation** → Median (episodes), mean (ratings), derived (seasons)
4. **Feature Engineering** → `avg_rating`, `anime_age`, `char_count`, `is_ongoing`
5. **Encoding** → Label Encoding (status, studio) + One-Hot (genre)
6. **Scaling** → Z-score (`StandardScaler`) + MinMax (`MinMaxScaler`)
7. **Validation** → 10 automated quality checks — all must PASS
8. **Export** → 3 purpose-built CSV outputs

## 📁 Output Files
| File                       | Rows | Cols | Use                   |
|------                      |------|------|-----                  |
| `anime_clean_master.csv`   | 97   | 39   | EDA & visualization   |
| `anime_clean_seasons.csv`  | 207  | 7    | Season-level analysis |
| `anime_ml_ready.csv`       | 97   | 32   |Direct sklearn input   |

## 🛠️ Tech Stack
`Python` · `pandas` · `numpy` · `scikit-learn`

## 🚀 Run It
```bash
python anime_data_pipeline.py
```
