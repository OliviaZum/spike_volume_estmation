import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull, cKDTree
import os
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, spearmanr

import open3d as o3d
import numpy as np
import matplotlib.patches as mpatches
from pathlib import Path
import seaborn as sns

#pcd = o3d.io.read_point_cloud("/projects/zumstego/1_datasets/scans_volumes/7_7_3.ply")
#pts = np.asarray(pcd.points)

#print("min xyz:", pts.min(axis=0))
#print("max xyz:", pts.max(axis=0))
#print("range   :", pts.ptp(axis=0))

model = "area" #NN / geo / area


# --- Load dictionaries of test data / predictions / geometric features from ply files ---
if model == "NN": 
    file_path = Path("data/out_NN")
    #neural networks
    with open("data/2_fine-tuned-mlp/1_dinov3-6imgs_mlp/cnn_run_0/mapping_test.json") as f: #all predictions: train / test / val
        true_dict = json.load(f)
    with open("data/2_fine-tuned-mlp/1_dinov3-6imgs_mlp/cnn_run_0/results/predictions.json") as f:
        pred_dict = json.load(f)

    # Build lookup: REAL plant_id → pred_entry
    pred_by_plant_id = {
        p["plant_id"]: p for p in pred_dict.values()
    }
    merged = {}
    for plant_id, true_entry in true_dict.items():

        if plant_id not in pred_by_plant_id:
            print("Missing prediction for:", plant_id)
            continue
        pred = pred_by_plant_id[plant_id]

        merged[plant_id] = {
            **true_entry,
            "volume_true": true_entry["volume"],
            "volume_predicted": pred["volume"]
        }
        del merged[plant_id]["volume"]  # remove old True volume field

    #print(merged)
    #save
    full_path = file_path / "merged_full_NN.json"
    with open(full_path, "w") as f:
        json.dump(merged, f, indent=2)
    df = pd.DataFrame(merged.values())

    df = pd.DataFrame([
    {
        "plant_id": pid,
        "volume_true": entry["volume_true"],
        "volume_predicted": entry["volume_predicted"],
        "genotype_id" : entry["genotype_id"]
    }
    for pid, entry in merged.items()
    ])


elif model == "geo": 
    file_path = Path("data/out_geometric_test")
    #geometric baseline
    with open("data/1_dinov2_6imgs_lstm/cnn_run_0/mapping_test.json") as f:
        true_dict = json.load(f)
    df_pred = pd.read_csv("data/out_geometric_test/merged_6_imgs.csv") #all predictions

    df_pred = df_pred.rename(columns={"Plant_id": "plant_id", 
                                      "Volume_true": "volume_true", 
                                      "volume_predicted_scaled": "volume_predicted"})
    
    #filter test set
    merged = {}
    for plant_id, true_entry in true_dict.items():

        # ensure this plant_id exists in the prediction CSV
        row = df_pred.loc[df_pred["plant_id"] == plant_id]

        if row.empty:
            print("Missing prediction for:", plant_id)
            continue

        row = row.iloc[0]

        merged[plant_id] = {
            "plant_id": plant_id,
            "volume_true": row["volume_true"],           # from CSV
            "volume_predicted": row["volume_predicted"], # from CSV
            "genotype_id": true_entry["genotype_id"]     # from test mapping
        }
        
        full_path = file_path / "merged_full_geo.json"
        with open(full_path, "w") as f:
            json.dump(merged, f, indent=2)
        df = pd.DataFrame(merged.values())

elif model == "area": 
    
    file_path = Path("data/out_area_test")
    #geometric baseline
    with open("data/1_dinov2_6imgs_lstm/cnn_run_0/mapping_test.json") as f:
        true_dict = json.load(f)
    df_pred = pd.read_csv("data/out_area_test/merged_6_imgs.csv") #all predictions

    df_pred = df_pred.rename(columns={"Plant_id": "plant_id", 
                                      "Volume_true": "volume_true", 
                                      "volume_predicted_scaled": "volume_predicted"})
    
    #filter test set
    merged = {}
    for plant_id, true_entry in true_dict.items():

        # ensure this plant_id exists in the prediction CSV
        row = df_pred.loc[df_pred["plant_id"] == plant_id]

        if row.empty:
            print("Missing prediction for:", plant_id)
            continue

        row = row.iloc[0]

        merged[plant_id] = {
            "plant_id": plant_id,
            "volume_true": row["volume_true"],           # from CSV
            "volume_predicted": row["volume_predicted"], # from CSV
            "genotype_id": true_entry["genotype_id"]     # from test mapping
        }
        
        full_path = file_path / "merged_full_area.json"
        with open(full_path, "w") as f:
            json.dump(merged, f, indent=2)
        df = pd.DataFrame(merged.values())



#####################################################################
#per spike data frame 
#add absolute error and squared error 
df["error"] = df["volume_predicted"] - df["volume_true"]
df["abs_error"] = df["error"].abs()
df["squared_error"] = df["error"]**2
df["mape"] = df["abs_error"] / df["volume_true"]

#add geometric features
geo_features = pd.read_excel("scans_volumes_traits.xlsx")
geo_df_test = geo_features[[
    "file_id",
    "length",
    "width",
    "inclination",
    "curvature"
]]
geo_df_test = geo_df_test.rename(columns={"file_id":"plant_id"})
geo_df_test["length"] = geo_df_test["length"] * 1000
geo_df_test["width"] = geo_df_test["width"] * 1000

#merge geometric features
df = df.merge(geo_df_test, on="plant_id", how="left")

print(df[df["curvature"].isna()]) # 3 removed: 8_8_3 / 5_18_9 / 18_18_5
print(df[df["length"].isna()])# same as above

print(len(df))
df = df.dropna(subset=["curvature", "length"])
print(len(df))

print(df[df["curvature"].isna()]) # 0
print(df[df["length"].isna()])# 0

#remove unrealistic measurements: 5
#df = df[
#    (df["curvature"] <= 2000) 
#]

print(len(df))


####################################################################
#genotype groups

group_stats = df.groupby("genotype_id").agg(
    n_samples=("plant_id", "count"),
    mae=("abs_error", "mean"),
    mape=("mape", "mean"), 
    rmse=("squared_error", lambda x: np.sqrt(x.mean())),
    bias=("error", "mean"),
    true_std=("volume_true", "std"),
    pred_std=("volume_predicted", "std"),
    mean_true=("volume_true", "mean"),
    mean_pred=("volume_predicted", "mean"), 
    mean_length = ("length", "mean"),
    mean_width = ("width", "mean"),
    mean_curvature = ("curvature", "mean"),
    mean_inclination = ("inclination", "mean")

).reset_index()

#group_stats["std_norm_mape"] = group_stats["mape"] / group_stats["true_std"]
#print(group_stats)

#ranked = group_stats.sort_values("std_norm_mape", ascending=False)
ranked = group_stats.sort_values("mape", ascending=False)

if model == "NN": 
    full_path_ranked = file_path / "cnn_ranked_test.csv"
    ranked.to_csv(full_path_ranked, index=False)
elif model == "geo": 
    full_path_ranked = file_path / "geo_ranked_test.csv"
    ranked.to_csv(full_path_ranked, index=False)
elif model == "area": 
    full_path_ranked = file_path / "area_ranked_test.csv"
    ranked.to_csv(full_path_ranked, index=False)

####################################################################
#based on genotypes, group the genotypes and the single spikes

# have a look at the three groups with different prediction accuracy
# GENOTYPE GROUPING
q33_g = group_stats["mape"].quantile(0.33)
q66_g = group_stats["mape"].quantile(0.66)

def categorize(row):
    if row > q66_g:
        return "low"
    elif row > q33_g:
        return "medium"
    else:
        return "good"
    
# SPIKE GROUPING
q33_s = df["abs_error"].quantile(0.33)
q66_s = df["abs_error"].quantile(0.66)
    
def categorize_spike(e):
    if e > q66_s:
        return "low"     # bad
    elif e > q33_s:
        return "medium"
    else:
        return "good"


group_stats["mape_group"] = group_stats["mape"].apply(categorize)

# enforce ordering for x-axis
order = ["good", "medium", "low"]
group_stats["mape_group"] = pd.Categorical(
    group_stats["mape_group"], categories=order, ordered=True
)

# merge group + true_std ONCE
df_merged = df.merge(
    group_stats[["genotype_id", "mape_group", "true_std"]],
    on="genotype_id",
    how="left"
)


#add spike grouping 
df_merged["spike_error_group"] = df_merged["abs_error"].apply(categorize_spike)


if model == "NN": 
    full_path_merged = file_path / "cnn_merged.csv"
    df_merged.to_csv(full_path_merged, index=False)
elif model == "geo": 
    full_path_merged = file_path / "geo_merged.csv"
    df_merged.to_csv(full_path_merged, index=False)
elif model == "area": 
    full_path_merged = file_path / "area_merged.csv"
    df_merged.to_csv(full_path_merged, index=False)



#print(df_merged.columns.tolist())

###################################################################


#Across-genotype variability (std of genotype means): 630.3166006619052
#4692.011990658165

spikes_variability_std = df["volume_true"].std()
spikes_variability_mean = df["volume_true"].mean()

print("Mean (mean of all spikes):",
      spikes_variability_mean)

print("Variability (std of all spikes):",
      spikes_variability_std)

# Variability of genotype means (across-genotype variability)
group_means = df.groupby("genotype_id")["volume_true"].mean()
across_genotype_variability_std = group_means.std()

print("Std of genotype means:",
      across_genotype_variability_std)

group_std = df.groupby("genotype_id")["volume_true"].std()
across_genotype_mean_std = group_std.mean()

print("Mean of genotype std:",
      across_genotype_mean_std)




###################################################################
#spike-level plotting
###################################################################

# ----------------------------------
# SPIKE-LEVEL SCATTERPLOTS
# ----------------------------------


colors_spike = {"good": "black", "medium": "black", "low": "black"}

traits_spike = [
    ("volume_true", "True Volume [mm³]"),
    ("length", "Length [mm]"),
    ("width", "Width [mm]"),
    ("curvature", "Curvature [a.u.]")
    #("inclination", "Inclination [°]")
]

fig, axes = plt.subplots(2, 2, figsize=(18, 15))
axes = axes.flatten()

for ax, (trait, title) in zip(axes, traits_spike):

    # Plot spike-level scatter, colored by spike-level error group
    for group in ["good", "medium", "low"]:
        subset = df_merged[df_merged["spike_error_group"] == group]
        ax.scatter(
            subset[trait],
            subset["abs_error"],
            s=8, alpha=1, color=colors_spike[group],
            label=group if trait == "length" else ""
        )

    # Regression line
    X = df_merged[[trait]].values.reshape(-1, 1)
    y = df_merged["abs_error"].values
    lr = LinearRegression().fit(X, y)
    x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    ax.plot(x_line, lr.predict(x_line), color="black", linewidth=2)

    # Correlations
    pear_r, pear_p = pearsonr(df_merged[trait], df_merged["abs_error"])
    spear_rho, spear_p = spearmanr(df_merged[trait], df_merged["abs_error"])

    ax.text(
        0.05, 0.95,
        f"r: {pear_r:.2f}",
        #f"Pearson r={pear_r:.3f} (p={pear_p:.3g})",
        #f"Spearman ρ={spear_rho:.3f} (p={spear_p:.3g})",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=17,
        bbox=dict(boxstyle="round", facecolor="white", alpha=1)
    )

    ax.set_xlabel(title, fontsize = 30)
    ax.set_ylabel("Absolute Error [mm³]", fontsize =30)
    #ax.set_title(f"Abs Error vs {title}", fontsize = 26)
    ax.tick_params(axis='both', which='major', labelsize=18)

#fig.delaxes(axes[-1])
#fig.delaxes(axes[-2])


plt.tight_layout()

if model == "NN": 
    full_path_sc_spikes = file_path / "scatterplots_spikes_NN.png"
    plt.savefig(full_path_sc_spikes, dpi=300)
elif model == "geo": 
    full_path_sc_spikes = file_path / "scatterplots_spikes_geo.png"
    plt.savefig(full_path_sc_spikes, dpi=300)
elif model == "area": 
    full_path_sc_spikes = file_path / "scatterplots_spikes_area.png"
    plt.savefig(full_path_sc_spikes, dpi=300)

###################################################################
#spike-level plotting
###################################################################

# ----------------------------------
# SPIKE-LEVEL SCATTERPLOTS
# ----------------------------------


colors_spike = {"good": "black", "medium": "black", "low": "black"}

traits_spike = [
    ("volume_true", "True Volume [mm³]"),
    ("length", "Length [mm]"),
    ("width", "Width [mm]"),
    ("curvature", "Curvature [a.u.]")
    #("inclination", "Inclination [°]")
]

fig, axes = plt.subplots(2, 2, figsize=(18, 15))
axes = axes.flatten()

for ax, (trait, title) in zip(axes, traits_spike):

    # Plot spike-level scatter, colored by spike-level error group
    for group in ["good", "medium", "low"]:
        subset = df_merged[df_merged["spike_error_group"] == group]
        ax.scatter(
            subset[trait],
            subset["error"],
            s=8, alpha=1, color=colors_spike[group],
            label=group if trait == "length" else ""
        )

    # Regression line
    X = df_merged[[trait]].values.reshape(-1, 1)
    y = df_merged["error"].values
    lr = LinearRegression().fit(X, y)
    x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    ax.plot(x_line, lr.predict(x_line), color="black", linewidth=2)

    # Correlations
    pear_r, pear_p = pearsonr(df_merged[trait], df_merged["error"])
    spear_rho, spear_p = spearmanr(df_merged[trait], df_merged["error"])

    ax.text(
        0.05, 0.95,
        #f"Pearson r={pear_r:.3f} (p={pear_p:.3g})",
        f"r: {pear_r:.2f}",
        #f"Spearman ρ={spear_rho:.3f} (p={spear_p:.3g})",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=40,
        bbox=dict(boxstyle="round", facecolor="white", alpha=1)
    )


    ax.set_xlabel(title, fontsize = 50)
    ax.set_ylabel("Error [mm³]", fontsize = 50)
    ax.set_ylim(top=2500)
    #ax.set_title(f"Abs Error vs {title}", fontsize = 16)
    ax.tick_params(axis='both', which='major', labelsize=35)

#fig.delaxes(axes[-1])
#fig.delaxes(axes[-2])

plt.tight_layout()

if model == "NN": 
    full_path_sc_spikes = file_path / "scatterplots_error_spikes_NN.png"
    plt.savefig(full_path_sc_spikes, dpi=900)
elif model == "geo": 
    full_path_sc_spikes = file_path / "scatterplots_error_spikes_geo.png"
    plt.savefig(full_path_sc_spikes, dpi=900)
elif model == "area": 
    full_path_sc_spikes = file_path / "scatterplots_error_spikes_area.png"
    plt.savefig(full_path_sc_spikes, dpi=900)
# ----------------------------------
# SPIKE-LEVEL BOXPLOTS (geometry by spike-level error group)
# ----------------------------------

order = ["good", "medium", "low"]  # spike-level

boxplot_traits = [
    ("volume_true", "True Volume [mm³]"),
    ("volume_predicted", "Predicted Volume [mm³]"),
    ("length", "Length [mm]"),
    ("width", "Width [mm]"),
    ("curvature", "Curvature [a.u.]"),
    ("inclination", "Inclination [°]")
]

fig, axes = plt.subplots(len(boxplot_traits), 1, figsize=(10, 36))
axes = axes.flatten()

for ax, (trait, title) in zip(axes, boxplot_traits):

    # FIX: use spike_error_group, not mape_group
    data = [
        df_merged[df_merged["spike_error_group"] == group][trait]
        for group in order
    ]

    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.set_title(f"{title} per Spike Error Group")
    ax.set_ylabel(title)
    ax.tick_params(axis="x", rotation=20)

plt.tight_layout()


if model == "NN": 
    full_path_bx_spikes = file_path / "boxplots_spikes.png"
    plt.savefig(full_path_bx_spikes, dpi=300)
elif model == "geo": 
    full_path_bx_spikes = file_path / "boxplots_spikes.png"
    plt.savefig(full_path_bx_spikes, dpi=300)
elif model == "area": 
    full_path_bx_spikes = file_path / "boxplots_spikes.png"
    plt.savefig(full_path_bx_spikes, dpi=300)


###################################################################
#genotype-level plotting
###################################################################

# ----------------------------------
# CREATE ONE FIGURE WITH MULTIPLE BOXPLOTS
# ----------------------------------
#take spike-level values, filter by genotye, create boxplot

genotype_boxplot_traits = [
    ("volume_true", "True Volume [mm³]"),
    ("mape", "MAPE [%]")
    #("abs_error", "Absolute Error [mm³]"),
    #("length", "Length [mm]"),
    #("width", "Width [mm]"),
    #("curvature", "Curvature [a.u.]"),
    #("inclination", "Inclination [°]")
    
]

fig, axes = plt.subplots(len(genotype_boxplot_traits), 1, figsize=(14, 3 * len(genotype_boxplot_traits)))
axes = axes.flatten()

#genotypes = sorted(df_merged["genotype_id"].unique())
genotypes = group_stats.sort_values("mape")["genotype_id"].tolist()


for ax, (trait, title) in zip(axes, genotype_boxplot_traits):

    # create a list of arrays: one per genotype
    data = [df_merged[df_merged["genotype_id"] == g][trait] for g in genotypes]

    ax.boxplot(data, tick_labels=genotypes, showfliers=False)
    ax.set_title(f"{title} per Genotype")
    ax.set_ylabel(title)
    ax.tick_params(axis="x", rotation=90)

plt.tight_layout()
#print("Saved figure: genotype_variability_and_error_boxplots.png"

if model == "NN": 
    full_path_bx_gen_mape = file_path / "boxplots_gen_mape.png"
    plt.savefig(full_path_bx_gen_mape, dpi=300)
elif model == "geo": 
    full_path_bx_gen_mape = file_path / "boxplots_gen_mape.png"
    plt.savefig(full_path_bx_gen_mape, dpi=300)
elif model == "area": 
    full_path_bx_gen_mape = file_path / "boxplots_gen_mape.png"
    plt.savefig(full_path_bx_gen_mape, dpi=300)

####################################################################
# BOX PLOT DATA

# volume
true_box = [df_merged[df_merged["mape_group"] == g]["volume_true"]        for g in order]
pred_box = [df_merged[df_merged["mape_group"] == g]["volume_predicted"]   for g in order]

# geometric traits
length_box     = [df_merged[df_merged["mape_group"] == g]["length"]       for g in order]
width_box      = [df_merged[df_merged["mape_group"] == g]["width"]        for g in order]
curvature_box  = [df_merged[df_merged["mape_group"] == g]["curvature"]    for g in order]
inclination_box  = [df_merged[df_merged["mape_group"] == g]["inclination"]    for g in order]

####################################################################
# CREATE FIGURE WITH MULTIPLE BOXPLOTS

order = ["good", "medium", "low"]  # spike-level
boxplot_traits = [
    ("volume_true", "True Volume [mm³]"),
    ("length", "Length [mm]"),
    ("width", "Width [mm]"),
    ("curvature", "Curvature [a.u.]"),
    ("inclination", "Inclination [°]")
]

fig, axes = plt.subplots(len(boxplot_traits), 1, figsize=(10, 36))
axes = axes.flatten()

for ax, (trait, title) in zip(axes, boxplot_traits):

    data = [
        df_merged[df_merged["mape_group"] == group][trait]
        for group in order
    ]

    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.set_title(f"{title} per Genotype Quality Group")
    ax.set_ylabel(title)
    ax.tick_params(axis="x", rotation=20)

plt.tight_layout()

if model == "NN": 
    full_path_bx_gen = file_path / "boxplots_gen.png"
    plt.savefig(full_path_bx_gen, dpi=300)
elif model == "geo": 
    full_path_bx_gen = file_path / "boxplots_gen.png"
    plt.savefig(full_path_bx_gen, dpi=300)
elif model == "area": 
    full_path_bx_gen = file_path / "boxplots_gen.png"
    plt.savefig(full_path_bx_gen, dpi=300)

################ genotype-level scatter plots ##########################################

traits = [
    ("mean_true", "Mean True Volume [mm³]"),
    ("mean_length", "Mean Length [mm]"),
    ("mean_width", "Mean Width [mm]"),
    ("mean_curvature", "Mean Curvature [a.u.]"),
    ("mean_inclination", "Mean Inclination [°]")
]

colors = {"good": "black", "medium": "black", "low": "black"}

fig, axes = plt.subplots(2, 3, figsize=(15, 15))
axes = axes.flatten()

for ax, (trait, title) in zip(axes, traits):

    # --- plot per-group points ---
    for group in ["good", "medium", "low"]:
        subset = group_stats[group_stats["mape_group"] == group]
        ax.scatter(
            subset[trait],
            subset["mape"],
            label=group if trait == "mean_length" else "",
            color=colors[group],
            alpha=1,
        )

    # --- regression line (genotype-level!) ---
    # Regression line — using genotype-level means
    X = group_stats[[trait]].values
    y = group_stats["mape"].values
    lr = LinearRegression().fit(X, y)

    x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    ax.plot(x_line, lr.predict(x_line), color="black", linewidth=2)

    
    # --- correlation (also genotype-level) ---
    pear_r, pear_p = pearsonr(group_stats[trait], group_stats["mape"])
    spear_rho, spear_p = spearmanr(group_stats[trait], group_stats["mape"])

    ax.text(
        0.05, 0.95,
        #f"Pearson r={pear_r:.3f} (p={pear_p:.3g})",
        f"r: {pear_r:.2f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=1)
    )

    ax.set_xlabel(title)
    ax.set_ylabel("MAPE [%]")
    ax.set_title(f"MAPE vs {title}")

fig.delaxes(axes[-1])


plt.tight_layout()


if model == "NN": 
    full_path_sc_gen = file_path / "scatterplots_gen.png"
    plt.savefig(full_path_sc_gen, dpi=300)
elif model == "geo": 
    full_path_sc_gen = file_path / "scatterplots_gen.png"
    plt.savefig(full_path_sc_gen, dpi=300)
elif model == "area": 
    full_path_sc_gen = file_path / "scatterplots_gen.png"
    plt.savefig(full_path_sc_gen, dpi=300)


###################################################################
#additional plots
###################################################################

###################################################################
# FIGURE 1: Error vs spike volume groups
###################################################################

df_merged["volume_group"] = pd.qcut(
    df_merged["volume_true"],
    q=3,
    labels=["small", "medium", "large"]
)

plt.figure(figsize=(8,6))

sns.boxplot(
    data=df_merged,
    x="volume_group",
    y="abs_error",
    showfliers=False,
    color="lightgray"
)

sns.stripplot(
    data=df_merged,
    x="volume_group",
    y="abs_error",
    color="black",
    size=3,
    alpha=0.6
)

plt.xlabel("Spike Volume Group", fontsize=24)
plt.ylabel("Absolute Error [mm³]", fontsize=24)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)


plt.tight_layout()

if model == "NN":
    plt.savefig(file_path / "error_vs_volume_groups_NN.png", dpi=300)
elif model == "geo":
    plt.savefig(file_path / "error_vs_volume_groups_geo.png", dpi=300)
elif model == "area":
    plt.savefig(file_path / "error_vs_volume_groups_area.png", dpi=300)

###################################################################
# FIGURE 2: Error vs spike shape groups
###################################################################

df_merged["aspect_ratio"] = df_merged["length"] / df_merged["width"]

df_merged["shape_group"] = pd.qcut(
    df_merged["aspect_ratio"],
    q=3,
    labels=["short-thick", "intermediate", "long-narrow"]
)

plt.figure(figsize=(8,6))

sns.boxplot(
    data=df_merged,
    x="shape_group",
    y="abs_error",
    showfliers=False,
    color="lightgray"
)

sns.stripplot(
    data=df_merged,
    x="shape_group",
    y="abs_error",
    color="black",
    size=3,
    alpha=0.6
)

plt.xlabel("Spike Shape Group", fontsize=24)
plt.ylabel("Absolute Error [mm³]", fontsize=24)


plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.tight_layout()

if model == "NN":
    plt.savefig(file_path / "error_vs_shape_groups_NN.png", dpi=300)
elif model == "geo":
    plt.savefig(file_path / "error_vs_shape_groups_geo.png", dpi=300)
elif model == "area":
    plt.savefig(file_path / "error_vs_shape_groups_area.png", dpi=300)

###################################################################
# FIGURE 3: Error vs curvature groups
###################################################################

df_merged["curvature_group"] = pd.qcut(
    df_merged["curvature"],
    q=3,
    labels=["low", "medium", "high"]
)

plt.figure(figsize=(8,6))

sns.boxplot(
    data=df_merged,
    x="curvature_group",
    y="abs_error",
    showfliers=False,
    color="lightgray"
)

sns.stripplot(
    data=df_merged,
    x="curvature_group",
    y="abs_error",
    color="black",
    size=3,
    alpha=0.6
)

plt.xlabel("Curvature Group", fontsize=24)
plt.ylabel("Absolute Error [mm³]", fontsize=24)


plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.tight_layout()

if model == "NN":
    plt.savefig(file_path / "error_vs_curvature_groups_NN.png", dpi=300)
elif model == "geo":
    plt.savefig(file_path / "error_vs_curvature_groups_geo.png", dpi=300)
elif model == "area":
    plt.savefig(file_path / "error_vs_curvature_groups_area.png", dpi=300)




###################################################################
#genotype-level plotting
###################################################################

#make a joint plot, genotype-specific true volume and mape:
cols_to_keep = ["plant_id", "volume_true", "volume_predicted", "genotype_id", "mape", "abs_error"] 

df_pred_geo = pd.read_csv("data/out_geometric_test/geo_merged.csv") #all predictions
df_pred_geo = df_pred_geo[cols_to_keep]
df_pred_geo = df_pred_geo.rename(columns={
    col: col + "_geo" for col in df_pred_geo.columns if col not in ["plant_id", "genotype_id"]
})

print(df_pred_geo)

df_pred_NN = pd.read_csv("data/out_NN/cnn_merged.csv") #all predictions
df_pred_NN = df_pred_NN[cols_to_keep]
df_pred_NN = df_pred_NN.rename(columns={
    col: col + "_NN" for col in df_pred_NN.columns if col not in ["plant_id", "genotype_id"]
})
print(df_pred_NN)

df_pred_area = pd.read_csv("data/out_area_test/area_merged.csv") #all predictions
df_pred_area = df_pred_area[cols_to_keep]
df_pred_area = df_pred_area.rename(columns={
    col: col + "_area" for col in df_pred_area.columns if col not in ["plant_id", "genotype_id"]
})
print(df_pred_area)

#merge_dfs
df_merge_geo_NN = pd.merge(df_pred_geo, df_pred_NN, on=["plant_id", "genotype_id"], how="inner")
df_merged_all = pd.merge(df_merge_geo_NN, df_pred_area, on=["plant_id", "genotype_id"], how="inner")
print(df_merged_all)
print()
#df_merged_all.to_csv("merged_check.csv", index=False)


df_merged_all["error_geo"] = (
    df_merged_all["volume_predicted_geo"] -
    df_merged_all["volume_true_geo"]
)

df_merged_all["error_area"] = (
    df_merged_all["volume_predicted_area"] -
    df_merged_all["volume_true_area"]
)

df_merged_all["error_NN"] = (
    df_merged_all["volume_predicted_NN"] -
    df_merged_all["volume_true_NN"]
)

# -------------------------------------------------------
# ADD GEOMETRIC TRAITS (needed for shape analysis)
# -------------------------------------------------------

geo_features = pd.read_excel("scans_volumes_traits.xlsx")

geo_df = geo_features[
    ["file_id", "length", "width", "curvature", "inclination"]
]

geo_df = geo_df.rename(columns={"file_id": "plant_id"})

# convert to mm
geo_df["length"] = geo_df["length"] * 1000
geo_df["width"] = geo_df["width"] * 1000

# merge with prediction dataframe
df_merged_all = df_merged_all.merge(
    geo_df,
    on="plant_id",
    how="left"
)


###################################################################
#genotype-level plotting
###################################################################

genotype_order = (
    df_merged_all.groupby("genotype_id")["volume_true_NN"]
    .mean()
    .sort_values()
)

# Genotype order (sort by NN, or GEO, or true volume)
#genotypes = sorted(df_merged_all["genotype_id"].unique())
genotypes = list(genotype_order.index)
print(genotypes)


# Collect GEO and NN values per genotype
ae_geo_data = [df_merged_all[df_merged_all["genotype_id"] == g]["abs_error_geo"] for g in genotypes]
ae_nn_data  = [df_merged_all[df_merged_all["genotype_id"] == g]["abs_error_NN"] for g in genotypes]
ae_area_data  = [df_merged_all[df_merged_all["genotype_id"] == g]["abs_error_area"] for g in genotypes]


plt.figure(figsize=(16, 6))
x = np.arange(len(genotypes))

# Three positions per genotype
positions_geo  = x - 0.25
positions_area = x
positions_nn   = x + 0.25

# GEO boxplots
plt.boxplot(
    ae_geo_data,
    positions=positions_geo,
    widths=0.22,
    showfliers=False,
    patch_artist=True,
    boxprops=dict(facecolor="lightblue"),
    tick_labels=[None] * len(genotypes)
)

# AREA boxplots
plt.boxplot(
    ae_area_data,
    positions=positions_area,
    widths=0.22,
    showfliers=False,
    patch_artist=True,
    boxprops=dict(facecolor="lightgrey"),
    tick_labels=[None] * len(genotypes)
)

# NN boxplots
plt.boxplot(
    ae_nn_data,
    positions=positions_nn,
    widths=0.22,
    showfliers=False,
    patch_artist=True,
    boxprops=dict(facecolor="lightgreen"),
    tick_labels=[None] * len(genotypes)
)

# Shared X-axis ticks
plt.xticks(x, genotypes, rotation=90)
plt.ylabel("MAE [mm³]")
#plt.title("MAE per Genotype: GEO vs AREA vs NN")

# Legend
plt.legend(
    handles=[
        mpatches.Patch(color="lightblue",  label="Geometric Baseline"),
        mpatches.Patch(color="lightgrey",  label="Area Baseline"),
        mpatches.Patch(color="lightgreen", label="MLP (DINOv3)")
    ]
)

plt.tight_layout()
plt.savefig(file_path / "boxplot_geo_area_nn_mape.png", dpi=300)


#plant height vs width: 
df = pd.read_csv("data/out_geometric_test/geo_merged.csv")

fig, ax = plt.subplots(figsize=(10, 8))

# Scatter
ax.scatter(df["length"], df["width"], s=12, alpha=1, color="black")

# Regression line
X = df[["length"]].values
y = df["width"].values
lr = LinearRegression().fit(X, y)
x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
ax.plot(x_line, lr.predict(x_line), color="black", linewidth=2)

# Correlation box
pear_r, pear_p = pearsonr(df["length"], df["width"])
ax.text(
    0.05, 0.95,
    f"r: {pear_r:.2f}",
    #f"Pearson r={pear_r:.3f} (p={pear_p:.3g})",
    transform=ax.transAxes,
    va="top",
    fontsize=25,
    bbox=dict(boxstyle="round", facecolor="white", alpha=1)
)

ax.set_xlabel("Length [mm]", fontsize=30)
ax.set_ylabel("Width [mm]", fontsize=30)
ax.tick_params(axis="both", labelsize=12)
ax.tick_params(axis='both', which='major', labelsize=25)


plt.tight_layout()
plt.savefig(file_path / "length_width.png", dpi=900)


################################################
#additional plots. 













################################################
# FIGURE 4: Model comparison across TRUE
# morphological groups (2D definition)
################################################

# ------------------------------------------------
# spike-level thresholds (25% tails)
# ------------------------------------------------

length_low  = df_merged_all["length"].quantile(0.25)
length_high = df_merged_all["length"].quantile(0.75)

width_low   = df_merged_all["width"].quantile(0.25)
width_high  = df_merged_all["width"].quantile(0.75)

# ------------------------------------------------
# create 2D morphological groups
# ------------------------------------------------

df_merged_all["shape_group"] = "intermediate"

# short-thick
df_merged_all.loc[
    (df_merged_all["length"] <= length_low) &
    (df_merged_all["width"]  >= width_high),
    "shape_group"
] = "short-thick"

# long-thin
df_merged_all.loc[
    (df_merged_all["length"] >= length_high) &
    (df_merged_all["width"]  <= width_low),
    "shape_group"
] = "long-thin"

print("\nSpikes per morphological group:")
print(df_merged_all["shape_group"].value_counts())

# ------------------------------------------------
# enforce order
# ------------------------------------------------

shape_order = ["short-thick", "intermediate", "long-thin"]

df_merged_all["shape_group"] = pd.Categorical(
    df_merged_all["shape_group"],
    categories=shape_order,
    ordered=True
)

# ------------------------------------------------
# prepare plot dataframe (blockwise stacking)
# ------------------------------------------------

df_plot = pd.concat([
    pd.DataFrame({
        "shape_group": df_merged_all["shape_group"],
        "error": df_merged_all["error_geo"],
        "Model": "Geometric baseline"
    }),
    pd.DataFrame({
        "shape_group": df_merged_all["shape_group"],
        "error": df_merged_all["error_area"],
        "Model": "Area baseline"
    }),
    pd.DataFrame({
        "shape_group": df_merged_all["shape_group"],
        "error": df_merged_all["error_NN"],
        "Model": "MLP (DINOv3)"
    })
], ignore_index=True)

# ------------------------------------------------
# plot
# ------------------------------------------------

plt.figure(figsize=(10,6))

sns.boxplot(
    data=df_plot,
    x="shape_group",
    y="error",
    hue="Model",
    order=shape_order,
    showfliers=False
)

plt.axhline(0, color="red", linestyle="--", linewidth=0.8)

plt.xlabel("Spike Morphological Group", fontsize=24)
plt.ylabel("Error [mm³]", fontsize=24)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.legend(
    loc="upper left",
    bbox_to_anchor=(0.02, 0.98),
    fontsize=18,
    title="Model",
    title_fontsize=20,
    frameon=True,
    facecolor="white",
    framealpha=0.8,
    edgecolor="black"
)

plt.tight_layout()

plt.savefig(
    file_path / "model_comparison_morphological_groups.png",
    dpi=1200
)





################################################
# FIGURE 4: TRUE morphological size groups
# (short–thin / intermediate / long–thick)
################################################

# ------------------------------------------------
# compute spike-level thresholds (25% tails)
# ------------------------------------------------

length_low  = df_merged_all["length"].quantile(0.25)
length_high = df_merged_all["length"].quantile(0.75)

width_low   = df_merged_all["width"].quantile(0.25)
width_high  = df_merged_all["width"].quantile(0.75)

# ------------------------------------------------
# create true morphological size groups (2D)
# ------------------------------------------------

df_merged_all["size_group"] = "intermediate"

# TRUE short-thin
df_merged_all.loc[
    (df_merged_all["length"] <= length_low) &
    (df_merged_all["width"]  <= width_low),
    "size_group"
] = "short-thin"

# TRUE long-thick
df_merged_all.loc[
    (df_merged_all["length"] >= length_high) &
    (df_merged_all["width"]  >= width_high),
    "size_group"
] = "long-thick"

print("\nSpikes per TRUE size group:")
print(df_merged_all["size_group"].value_counts())

# ------------------------------------------------
# enforce order (3 groups)
# ------------------------------------------------

size_order = ["short-thin", "intermediate", "long-thick"]

df_merged_all["size_group"] = pd.Categorical(
    df_merged_all["size_group"],
    categories=size_order,
    ordered=True
)

# ------------------------------------------------
# prepare dataframe for plotting
# ------------------------------------------------

df_plot = pd.concat([
    pd.DataFrame({
        "size_group": df_merged_all["size_group"],
        "error": df_merged_all["error_geo"],
        "Model": "Geometric baseline"
    }),
    pd.DataFrame({
        "size_group": df_merged_all["size_group"],
        "error": df_merged_all["error_area"],
        "Model": "Area baseline"
    }),
    pd.DataFrame({
        "size_group": df_merged_all["size_group"],
        "error": df_merged_all["error_NN"],
        "Model": "MLP (DINOv3)"
    })
], ignore_index=True)

# ------------------------------------------------
# plot
# ------------------------------------------------

plt.figure(figsize=(10,6))

sns.boxplot(
    data=df_plot,
    x="size_group",
    y="error",
    hue="Model",
    order=size_order,
    showfliers=False
)

plt.axhline(0, color="red", linestyle="--", linewidth=0.8)

plt.xlabel("Spike Morphological Group", fontsize=24)
plt.ylabel("Error [mm³]", fontsize=24)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.legend(
    loc="upper left",
    bbox_to_anchor=(0.02, 0.98),
    fontsize=18,
    title="Model",
    title_fontsize=20,
    frameon=True,
    facecolor="white",
    framealpha=0.8,
    edgecolor="black"
)

plt.tight_layout()

plt.savefig(
    file_path / "model_comparison_morphological_size_groups.png",
    dpi=1200
)








###################################################################
# FIGURE 5: Model comparison across spike volume groups
# (20% / 60% / 20% based on SPIKES)
###################################################################

# ------------------------------------------------
# spike-level thresholds (20% / 80%)
# ------------------------------------------------

low = df_merged_all["volume_true_NN"].quantile(0.15)
high = df_merged_all["volume_true_NN"].quantile(0.85)

# ------------------------------------------------
# create spike-level groups
# ------------------------------------------------

df_merged_all["volume_group"] = "intermediate"

df_merged_all.loc[
    df_merged_all["volume_true_NN"] <= low,
    "volume_group"
] = "small"

df_merged_all.loc[
    df_merged_all["volume_true_NN"] >= high,
    "volume_group"
] = "large"

print("\nSpikes per volume group:")
print(df_merged_all["volume_group"].value_counts())

# ------------------------------------------------
# enforce order
# ------------------------------------------------

volume_order = ["small", "intermediate", "large"]

df_merged_all["volume_group"] = pd.Categorical(
    df_merged_all["volume_group"],
    categories=volume_order,
    ordered=True
)

# ------------------------------------------------
# prepare dataframe for plotting
# ------------------------------------------------

df_plot = pd.concat([
    pd.DataFrame({
        "volume_group": df_merged_all["volume_group"],
        "error": df_merged_all["error_geo"],
        "Model": "Geometric baseline"
    }),
    pd.DataFrame({
        "volume_group": df_merged_all["volume_group"],
        "error": df_merged_all["error_area"],
        "Model": "Area baseline"
    }),
    pd.DataFrame({
        "volume_group": df_merged_all["volume_group"],
        "error": df_merged_all["error_NN"],
        "Model": "MLP (DINOv3)"
    })
], ignore_index=True)

# ------------------------------------------------
# plot
# ------------------------------------------------

plt.figure(figsize=(10,6))

sns.boxplot(
    data=df_plot,
    x="volume_group",
    y="error",
    hue="Model",
    order=volume_order,
    showfliers=False
)

plt.axhline(0, color="red", linestyle="--", linewidth=0.8)

plt.xlabel("Spike Volume Group", fontsize=24)
plt.ylabel("Error [mm³]", fontsize=24)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.legend(
    loc="upper left",
    bbox_to_anchor=(0.02, 0.98),
    fontsize=18,
    title="Model",
    title_fontsize=20,
    frameon=True,
    facecolor="white",
    framealpha=0.8,
    edgecolor="black"
)



plt.tight_layout()

plt.savefig(file_path / "model_comparison_volume_groups.png", dpi=1200)








#################################################################
# FIGURE: Model comparison across curvature groups
# (20% / 60% / 20% based on SPIKES)
#################################################################

# ------------------------------------------------
# spike-level thresholds (20% / 80%)
# ------------------------------------------------

low = df_merged_all["curvature"].quantile(0.15)
high = df_merged_all["curvature"].quantile(0.85)

# ------------------------------------------------
# create spike-level groups
# ------------------------------------------------

df_merged_all["curvature_group"] = "medium"

df_merged_all.loc[
    df_merged_all["curvature"] <= low,
    "curvature_group"
] = "low"

df_merged_all.loc[
    df_merged_all["curvature"] >= high,
    "curvature_group"
] = "high"

print("\nSpikes per curvature group:")
print(df_merged_all["curvature_group"].value_counts())

# ------------------------------------------------
# enforce order
# ------------------------------------------------

curvature_order = ["low", "medium", "high"]

df_merged_all["curvature_group"] = pd.Categorical(
    df_merged_all["curvature_group"],
    categories=curvature_order,
    ordered=True
)

# ------------------------------------------------
# prepare dataframe for plotting
# ------------------------------------------------

df_plot = pd.concat([
    pd.DataFrame({
        "curvature_group": df_merged_all["curvature_group"],
        "error": df_merged_all["error_geo"],
        "Model": "Geometric baseline"
    }),
    pd.DataFrame({
        "curvature_group": df_merged_all["curvature_group"],
        "error": df_merged_all["error_area"],
        "Model": "Area baseline"
    }),
    pd.DataFrame({
        "curvature_group": df_merged_all["curvature_group"],
        "error": df_merged_all["error_NN"],
        "Model": "MLP (DINOv3)"
    })
], ignore_index=True)

# ------------------------------------------------
# plot
# ------------------------------------------------

plt.figure(figsize=(10,6))

sns.boxplot(
    data=df_plot.dropna(subset=["curvature_group","error"]),
    x="curvature_group",
    y="error",
    hue="Model",
    order=curvature_order,
    showfliers=False
)

plt.axhline(0, color="red", linestyle="--", linewidth=0.8)

plt.xlabel("Curvature Group", fontsize=24)
plt.ylabel("Error [mm³]", fontsize=24)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.legend(
    loc="upper left",
    bbox_to_anchor=(0.02, 0.98),
    fontsize=18,
    title="Model",
    title_fontsize=20,
    frameon=True,
    facecolor="white",
    framealpha=0.8,
    edgecolor="black"
)



plt.tight_layout()

plt.savefig(file_path / "model_comparison_curvature_groups.png", dpi=1200)

###########################################################








from sklearn.metrics import r2_score

###################################################################
# FUNCTION TO COMPUTE METRICS
###################################################################

def compute_metrics(df, group_column):

    results = []

    models = {
        "Geometric baseline": ("volume_true_geo", "volume_predicted_geo"),
        "Area baseline": ("volume_true_area", "volume_predicted_area"),
        "MLP (DINOv3)": ("volume_true_NN", "volume_predicted_NN")
    }

    for group, subset in df.groupby(group_column):

        for model_name, (true_col, pred_col) in models.items():

            y_true = subset[true_col].values
            y_pred = subset[pred_col].values

            mae = np.mean(np.abs(y_pred - y_true))
            mape = np.mean(np.abs((y_pred - y_true) / y_true)) * 100
            r2 = r2_score(y_true, y_pred)

            corr = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else np.nan

            results.append({
                "group": group,
                "model": model_name,
                "corr": corr,
                "r2": r2,
                "mape": mape,
                "mae": mae
            })

    return pd.DataFrame(results)


###################################################################
# STRATIFIED METRICS TABLES
###################################################################

volume_metrics = compute_metrics(df_merged_all, "volume_group")
shape_metrics = compute_metrics(df_merged_all, "shape_group")
curvature_metrics = compute_metrics(df_merged_all, "curvature_group")
size_metrics = compute_metrics(df_merged_all, "size_group")

print(volume_metrics)
print(shape_metrics)
print(curvature_metrics)
print(size_metrics)


#####################################################################

#add geometric features
traits_all_spikes = pd.read_excel("scans_volumes_traits.xlsx")
traits_all_spikes["length"] = traits_all_spikes["length"] * 1000
traits_all_spikes["width"] = traits_all_spikes["width"] * 1000

print(traits_all_spikes)

stats = traits_all_spikes[["length", "width", "curvature"]].agg(["mean", "std"])
print(stats)








########################################
#plot the groups: 
###################################################################
################################################
# HISTOGRAM: Volume distribution + extreme groups
################################################

################################################
# HISTOGRAM: Volume distribution + TRUE group thresholds
################################################
################################################
# HISTOGRAM — EXACT SAME SPIKES AS FIGURE 5
################################################

################################################
# HISTOGRAM — with dotted group boundaries
################################################

vol = df_merged_all["volume_true_NN"]

# get boundaries DIRECTLY from your groups
max_small = df_merged_all[df_merged_all["volume_group"] == "small"]["volume_true_NN"].max()
min_large = df_merged_all[df_merged_all["volume_group"] == "large"]["volume_true_NN"].min()

plt.figure(figsize=(10, 6))

# --- grey histogram
plt.hist(vol, bins=40, color="lightgray", edgecolor="black")

# --- dotted lines separating the 3 groups
plt.axvline(max_small, color="black", linestyle="--", linewidth=2)
plt.axvline(min_large, color="black", linestyle="--", linewidth=2)

plt.xlabel("True Volume [mm³]", fontsize=24)
plt.ylabel("Count", fontsize=24)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.tight_layout()
plt.savefig(file_path / "histogram_volume_groups_lines.png", dpi=1200)


#curvature
################################################
# HISTOGRAM — Curvature with group boundaries
################################################

curv = df_merged_all["curvature"]

# get boundaries FROM YOUR EXISTING GROUPS
max_low = df_merged_all.loc[
    df_merged_all["curvature_group"] == "low", "curvature"
].max()

min_high = df_merged_all.loc[
    df_merged_all["curvature_group"] == "high", "curvature"
].min()

plt.figure(figsize=(10, 6))

# --- grey histogram
plt.hist(curv, bins=40, color="lightgray", edgecolor="black")

# --- dotted lines separating groups
plt.axvline(max_low,  color="black", linestyle="--", linewidth=2)
plt.axvline(min_high, color="black", linestyle="--", linewidth=2)

plt.xlabel("Curvature [a.u.]", fontsize=24)
plt.ylabel("Count", fontsize=24)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.tight_layout()
plt.savefig(file_path / "histogram_curvature_groups.png", dpi=1200)

#legnth: 

################################################
# HISTOGRAM — Length with size-group boundaries
################################################
################################################
# SCATTER — Length vs Width with size groups
################################################

df_merged_all["size_group_4"] = "intermediate"

# short-thin
df_merged_all.loc[
    (df_merged_all["length"] <= length_low) &
    (df_merged_all["width"]  <= width_low),
    "size_group_4"
] = "short-thin"

# short-thick
df_merged_all.loc[
    (df_merged_all["length"] <= length_low) &
    (df_merged_all["width"]  >= width_high),
    "size_group_4"
] = "short-thick"

# long-thin
df_merged_all.loc[
    (df_merged_all["length"] >= length_high) &
    (df_merged_all["width"]  <= width_low),
    "size_group_4"
] = "long-thin"

# long-thick
df_merged_all.loc[
    (df_merged_all["length"] >= length_high) &
    (df_merged_all["width"]  >= width_high),
    "size_group_4"
] = "long-thick"

plt.figure(figsize=(10, 8))

colors = {
    "short-thin": "blue",
    "short-thick": "green",
    "long-thin": "orange",
    "long-thick": "red",
    "intermediate": "lightgray"
}

order = ["short-thin", "short-thick", "long-thin", "long-thick", "intermediate"]

for group in order:
    subset = df_merged_all[df_merged_all["size_group_4"] == group]
    plt.scatter(
        subset["length"],
        subset["width"],
        s=15,
        color=colors[group],
        label=group,
        alpha=1
    )

# --- boundaries (important!)
plt.axvline(length_low,  linestyle="--", color="black", linewidth=2)
plt.axvline(length_high, linestyle="--", color="black", linewidth=2)

plt.axhline(width_low,  linestyle="--", color="black", linewidth=2)
plt.axhline(width_high, linestyle="--", color="black", linewidth=2)

plt.xlabel("Length [mm]", fontsize=24)
plt.ylabel("Width [mm]", fontsize=24)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.legend(
    fontsize=16,
    frameon=True,
    facecolor="white",
    edgecolor="black"
)

plt.tight_layout()
plt.savefig(file_path / "scatter_length_width_4groups.png", dpi=1200)