import pandas as pd

# =========================================
# LOAD
# =========================================

competitions = pd.read_csv(
    "data/kaggle/Competitions.csv"
)

tags = pd.read_csv(
    "data/kaggle/Tags.csv"
)

competition_tags = pd.read_csv(
    "data/kaggle/CompetitionTags.csv"
)

# =========================================
# MERGE TAGS
# =========================================

merged = competition_tags.merge(
    tags,
    left_on="TagId",
    right_on="Id"
)

tag_map = merged.groupby(
    "CompetitionId"
)["Name"].apply(list).to_dict()

# =========================================
# BUILD FINAL DATASET
# =========================================

rows = []

for _, row in competitions.iterrows():

    cid = row["Id"]

    rows.append({

        "title": row.get("Title", ""),

        "slug": row.get("Slug", ""),

        "subtitle": row.get("Subtitle", ""),

        "tags": ", ".join(
            tag_map.get(cid, [])
        )
    })

final_df = pd.DataFrame(rows)

# =========================================
# CLEAN
# =========================================

final_df = final_df.dropna()

final_df = final_df[
    final_df["title"].str.len() > 3
]

# =========================================
# SAVE
# =========================================

final_df.to_csv(
    "data/kaggle/kaggle_competitions_clean.csv",
    index=False
)

print(final_df.head())

print("\n✅ DONE")