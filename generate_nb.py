import json
import sys

cells_data = [
    ("markdown", "# STAGE 1 — Messidor EDA + preprocessing\n\n## Cell 1 — Imports"),
    ("code", """import os
import random
import copy
import time
import json
import math

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import (
    Dataset,
    DataLoader,
    Subset
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

print("Packages loaded")"""),

    ("markdown", "## Cell 2 — Reproducibility"),
    ("code", """SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", DEVICE)"""),

    ("markdown", "## Cell 3 — Dataset path\nUse the exact uploaded CSV path for now."),
    ("code", """DATASET_PATH = Path(
    r"/mnt/data/messidor_features(1).csv"
)

if not DATASET_PATH.exists():
    # Fallback to local Windows path
    DATASET_PATH = Path(
        r"D:\RMS\messidor_features.csv"
    )
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            DATASET_PATH
        )

print(
    "Dataset:",
    DATASET_PATH
)"""),

    ("markdown", "## Cell 4 — Load dataset"),
    ("code", """df_raw = pd.read_csv(
    DATASET_PATH
)

print(
    "Dataset shape:",
    df_raw.shape
)

print("\\nColumns:")
print(
    df_raw.columns.tolist()
)

print("\\nFirst 5 rows:")
display(
    df_raw.head()
)"""),

    ("markdown", "## Cell 5 — Basic dataset inspection"),
    ("code", """print("===== DATASET INFORMATION =====")

print(
    df_raw.info()
)

print("\\nMissing values:")

display(
    df_raw.isnull().sum()
)

print("\\nDuplicate rows:")

print(
    df_raw.duplicated().sum()
)"""),

    ("markdown", "## Cell 6 — Clean dataset"),
    ("code", """df = df_raw.copy()

# Remove completely empty rows
df = df.dropna(
    axis=0,
    how="all"
)

# Remove completely empty columns
df = df.dropna(
    axis=1,
    how="all"
)

# Remove duplicate rows
df = df.drop_duplicates()

# Remove unnamed columns if present
df = df.loc[
    :,
    ~df.columns.astype(str)
    .str.startswith("Unnamed")
]

print(
    "Cleaned shape:",
    df.shape
)"""),

    ("markdown", "## Cell 7 — Verify target"),
    ("code", """TARGET_COLUMN = "Class"

if TARGET_COLUMN not in df.columns:

    raise ValueError(
        "Class column not found"
    )

df[TARGET_COLUMN] = (
    pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce"
    )
)

df = df.dropna(
    subset=[TARGET_COLUMN]
)

df[TARGET_COLUMN] = (
    df[TARGET_COLUMN]
    .astype(int)
)

print(
    "Target classes:",
    sorted(
        df[TARGET_COLUMN]
        .unique()
    )
)

print(
    "\\nClass distribution:"
)

print(
    df[TARGET_COLUMN]
    .value_counts()
    .sort_index()
)"""),

    ("markdown", "# EDA\n\n## Cell 8 — Statistical summary"),
    ("code", """print(
    "===== STATISTICAL SUMMARY ====="
)

display(
    df.describe().T
)"""),

    ("markdown", "## Cell 9 — Class distribution"),
    ("code", """class_counts = (
    df[TARGET_COLUMN]
    .value_counts()
    .sort_index()
)

print(
    class_counts
)

plt.figure(
    figsize=(7, 5)
)

plt.bar(
    ["Class 0", "Class 1"],
    class_counts.values
)

plt.xlabel("Class")
plt.ylabel("Number of samples")
plt.title(
    "Messidor Class Distribution"
)

plt.tight_layout()

plt.show()"""),

    ("markdown", "## Cell 10 — Missing-value analysis"),
    ("code", """missing = (
    df.isnull()
    .sum()
    .sort_values(
        ascending=False
    )
)

display(
    missing
)"""),

    ("markdown", "## Cell 11 — Feature distributions"),
    ("code", """feature_columns = [
    column
    for column in df.columns
    if column != TARGET_COLUMN
]

df[
    feature_columns
].hist(
    figsize=(16, 14),
    bins=30
)

plt.suptitle(
    "Messidor Feature Distributions",
    y=1.02
)

plt.tight_layout()

plt.show()"""),

    ("markdown", "## Cell 12 — Correlation matrix"),
    ("code", """plt.figure(
    figsize=(16, 13)
)

correlation = df.corr(
    numeric_only=True
)

sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0
)

plt.title(
    "Messidor Feature Correlation Matrix"
)

plt.tight_layout()

plt.show()"""),

    ("markdown", "## Cell 13 — Outlier analysis\nDon't automatically remove these outliers. For medical data, an unusual value can be meaningful."),
    ("code", """plt.figure(
    figsize=(18, 7)
)

df[
    feature_columns
].boxplot(
    rot=90
)

plt.title(
    "Messidor Feature Outlier Analysis"
)

plt.tight_layout()

plt.show()"""),

    ("markdown", "# Train/validation/test split\n\n## Cell 14 — Split\nUse 70% training, 15% validation, 15% test and stratify by `Class`."),
    ("code", """train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=SEED,
    stratify=df[TARGET_COLUMN]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_df[TARGET_COLUMN]
)

train_df = train_df.reset_index(
    drop=True
)

val_df = val_df.reset_index(
    drop=True
)

test_df = test_df.reset_index(
    drop=True
)

print(
    "Training:",
    train_df.shape
)

print(
    "Validation:",
    val_df.shape
)

print(
    "Test:",
    test_df.shape
)"""),

    ("markdown", "# Cell 15 — Verify class balance"),
    ("code", """for name, split in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Testing", test_df)
]:

    print(
        f"\\n{name}"
    )

    print(
        split[
            TARGET_COLUMN
        ]
        .value_counts()
        .sort_index()
    )"""),

    ("markdown", "# Cell 16 — Standardization\n**Important:** fit the scaler only on training data."),
    ("code", """X_train = train_df[
    feature_columns
].values.astype(
    np.float32
)

y_train = train_df[
    TARGET_COLUMN
].values.astype(
    np.int64
)

X_val = val_df[
    feature_columns
].values.astype(
    np.float32
)

y_val = val_df[
    TARGET_COLUMN
].values.astype(
    np.int64
)

X_test = test_df[
    feature_columns
].values.astype(
    np.float32
)

y_test = test_df[
    TARGET_COLUMN
].values.astype(
    np.int64
)


scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_val = scaler.transform(
    X_val
)

X_test = scaler.transform(
    X_test
)

print(
    "Training mean:",
    X_train.mean()
)

print(
    "Training std:",
    X_train.std()
)"""),

    ("markdown", "## Cell 17 — Save preprocessing"),
    ("code", """MESSIDOR_OUTPUT = Path(
    r"D:\RMS\Messidor_FL_Results"
)

MESSIDOR_OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)

np.save(
    MESSIDOR_OUTPUT /
    "X_train.npy",
    X_train
)

np.save(
    MESSIDOR_OUTPUT /
    "X_val.npy",
    X_val
)

np.save(
    MESSIDOR_OUTPUT /
    "X_test.npy",
    X_test
)

np.save(
    MESSIDOR_OUTPUT /
    "y_train.npy",
    y_train
)

np.save(
    MESSIDOR_OUTPUT /
    "y_val.npy",
    y_val
)

np.save(
    MESSIDOR_OUTPUT /
    "y_test.npy",
    y_test
)

print(
    "Preprocessed data saved"
)"""),

    ("markdown", "# STAGE 2 — Local MLP\nThis replaces ResNet50 because your Messidor file contains numerical features.\n\n## Cell 18 — Tensor dataset"),
    ("code", """class MessidorDataset(Dataset):

    def __init__(
        self,
        X,
        y
    ):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.long
        )

    def __len__(self):

        return len(self.y)

    def __getitem__(
        self,
        index
    ):

        return (
            self.X[index],
            self.y[index]
        )


train_dataset = MessidorDataset(
    X_train,
    y_train
)

val_dataset = MessidorDataset(
    X_val,
    y_val
)

test_dataset = MessidorDataset(
    X_test,
    y_test
)

print(
    len(train_dataset),
    len(val_dataset),
    len(test_dataset)
)"""),

    ("markdown", "## Cell 19 — DataLoaders"),
    ("code", """BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)"""),

    ("markdown", "## Cell 20 — MLP model"),
    ("code", """INPUT_SIZE = len(
    feature_columns
)

NUM_CLASSES = 2


class MessidorMLP(nn.Module):

    def __init__(
        self,
        input_size
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_size,
                128
            ),

            nn.BatchNorm1d(
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.30
            ),

            nn.Linear(
                128,
                64
            ),

            nn.BatchNorm1d(
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                NUM_CLASSES
            )
        )

    def forward(
        self,
        x
    ):

        return self.network(x)


local_model = MessidorMLP(
    INPUT_SIZE
).to(DEVICE)

print(
    local_model
)"""),

    ("markdown", "## Cell 21 — Evaluation function"),
    ("code", """def evaluate_tabular_model(
    model,
    loader
):

    model.eval()

    labels_all = []
    predictions_all = []
    probabilities_all = []

    total_loss = 0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():

        for X, y in loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            outputs = model(X)

            loss = criterion(
                outputs,
                y
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            batch_size = y.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

            labels_all.extend(
                y.cpu().numpy()
            )

            predictions_all.extend(
                predictions.cpu().numpy()
            )

            probabilities_all.extend(
                probabilities[:, 1]
                .cpu()
                .numpy()
            )

    labels_all = np.array(
        labels_all
    )

    predictions_all = np.array(
        predictions_all
    )

    probabilities_all = np.array(
        probabilities_all
    )

    return {

        "loss":
            total_loss /
            total_samples,

        "accuracy":
            accuracy_score(
                labels_all,
                predictions_all
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                labels_all,
                predictions_all
            ),

        "precision":
            precision_score(
                labels_all,
                predictions_all,
                zero_division=0
            ),

        "recall":
            recall_score(
                labels_all,
                predictions_all,
                zero_division=0
            ),

        "macro_f1":
            f1_score(
                labels_all,
                predictions_all,
                average="macro",
                zero_division=0
            ),

        "auc":
            roc_auc_score(
                labels_all,
                probabilities_all
            ),

        "labels":
            labels_all,

        "predictions":
            predictions_all,

        "probabilities":
            probabilities_all
    }"""),

    ("markdown", "## Cell 22 — Local training"),
    ("code", """LOCAL_EPOCHS = 100

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    local_model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=8
)

best_state = copy.deepcopy(
    local_model.state_dict()
)

best_accuracy = 0

local_history = []

for epoch in range(
    1,
    LOCAL_EPOCHS + 1
):

    local_model.train()

    train_correct = 0
    train_total = 0
    train_loss = 0

    for X, y in train_loader:

        X = X.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        outputs = local_model(X)

        loss = criterion(
            outputs,
            y
        )

        loss.backward()

        optimizer.step()

        train_loss += (
            loss.item()
            * y.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        train_correct += (
            predictions == y
        ).sum().item()

        train_total += y.size(0)

    val_metrics = evaluate_tabular_model(
        local_model,
        val_loader
    )

    train_accuracy = (
        train_correct /
        train_total
    )

    scheduler.step(
        val_metrics["accuracy"]
    )

    local_history.append({

        "epoch": epoch,

        "train_accuracy":
            train_accuracy,

        "val_accuracy":
            val_metrics[
                "accuracy"
            ],

        "val_f1":
            val_metrics[
                "macro_f1"
            ],

        "val_auc":
            val_metrics[
                "auc"
            ]
    })

    print(
        f"Epoch {epoch:03d} | "
        f"Train "
        f"{train_accuracy*100:.2f}% | "
        f"Val "
        f"{val_metrics['accuracy']*100:.2f}% | "
        f"F1 "
        f"{val_metrics['macro_f1']*100:.2f}%"
    )

    if (
        val_metrics["accuracy"]
        > best_accuracy
    ):

        best_accuracy = (
            val_metrics["accuracy"]
        )

        best_state = copy.deepcopy(
            local_model.state_dict()
        )

local_model.load_state_dict(
    best_state
)

print(
    "Best local validation:",
    f"{best_accuracy*100:.2f}%"
)"""),

    ("markdown", "## Cell 23 — Local final test"),
    ("code", """local_test = evaluate_tabular_model(
    local_model,
    test_loader
)

print(
    "LOCAL MLP TEST"
)

print(
    "Accuracy:",
    f"{local_test['accuracy']*100:.2f}%"
)

print(
    "Balanced Accuracy:",
    f"{local_test['balanced_accuracy']*100:.2f}%"
)

print(
    "Macro F1:",
    f"{local_test['macro_f1']*100:.2f}%"
)

print(
    "AUC:",
    f"{local_test['auc']:.4f}"
)

print(
    classification_report(
        local_test["labels"],
        local_test["predictions"],
        target_names=[
            "Class 0",
            "Class 1"
        ]
    )
)"""),

    ("markdown", "# STAGE 3 — FedAvg MLP\nNow we distribute the **training data only** between clients.\n\n## Cell 24 — Client partition\nStart with 5 IID clients."),
    ("code", """NUM_CLIENTS = 5
CLIENTS_PER_ROUND = 5

FED_ROUNDS = 50
LOCAL_EPOCHS_FED = 3


def create_iid_partitions(
    num_samples,
    num_clients,
    seed=42
):

    rng = np.random.default_rng(
        seed
    )

    indices = np.arange(
        num_samples
    )

    rng.shuffle(indices)

    partitions = np.array_split(
        indices,
        num_clients
    )

    return [
        list(partition)
        for partition in partitions
    ]


client_indices = create_iid_partitions(
    len(train_dataset),
    NUM_CLIENTS,
    SEED
)

for client_id, indices in enumerate(
    client_indices
):

    client_labels = y_train[
        indices
    ]

    print(
        "Client",
        client_id + 1,
        "samples:",
        len(indices),
        "classes:",
        np.bincount(
            client_labels,
            minlength=2
        )
    )"""),

    ("markdown", "## Cell 25 — Client loaders"),
    ("code", """client_loaders = []

for indices in client_indices:

    client_subset = Subset(
        train_dataset,
        indices
    )

    client_loader = DataLoader(
        client_subset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    client_loaders.append(
        client_loader
    )"""),

    ("markdown", "## Cell 26 — FedAvg aggregation"),
    ("code", """def fedavg(
    client_states,
    client_sizes
):

    total_samples = sum(
        client_sizes
    )

    global_state = copy.deepcopy(
        client_states[0]
    )

    for key in global_state:

        global_state[key] = (
            client_states[0][key]
            *
            client_sizes[0]
            /
            total_samples
        )

        for client_id in range(
            1,
            len(client_states)
        ):

            global_state[key] += (
                client_states[client_id][key]
                *
                client_sizes[client_id]
                /
                total_samples
            )

    return global_state"""),

    ("markdown", "## Cell 27 — Client local training"),
    ("code", """def train_fed_client(
    global_state,
    loader
):

    model = MessidorMLP(
        INPUT_SIZE
    ).to(DEVICE)

    model.load_state_dict(
        copy.deepcopy(
            global_state
        )
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    model.train()

    for epoch in range(
        LOCAL_EPOCHS_FED
    ):

        for X, y in loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(X)

            loss = criterion(
                outputs,
                y
            )

            loss.backward()

            optimizer.step()

    return (
        model.state_dict(),
        len(loader.dataset)
    )"""),

    ("markdown", "## Cell 28 — FedAvg training"),
    ("code", """global_model = MessidorMLP(
    INPUT_SIZE
).to(DEVICE)

fed_history = []

best_fed_state = copy.deepcopy(
    global_model.state_dict()
)

best_fed_accuracy = 0

for round_number in range(
    1,
    FED_ROUNDS + 1
):

    client_states = []
    client_sizes = []

    for client_id in range(
        CLIENTS_PER_ROUND
    ):

        state, size = train_fed_client(
            global_model.state_dict(),
            client_loaders[client_id]
        )

        client_states.append(
            state
        )

        client_sizes.append(
            size
        )

    aggregated_state = fedavg(
        client_states,
        client_sizes
    )

    global_model.load_state_dict(
        aggregated_state
    )

    val_metrics = (
        evaluate_tabular_model(
            global_model,
            val_loader
        )
    )

    fed_history.append({

        "round":
            round_number,

        "accuracy":
            val_metrics["accuracy"],

        "macro_f1":
            val_metrics["macro_f1"],

        "auc":
            val_metrics["auc"]
    })

    print(
        f"Round {round_number:03d} | "
        f"Accuracy "
        f"{val_metrics['accuracy']*100:.2f}% | "
        f"F1 "
        f"{val_metrics['macro_f1']*100:.2f}%"
    )

    if (
        val_metrics["accuracy"]
        > best_fed_accuracy
    ):

        best_fed_accuracy = (
            val_metrics["accuracy"]
        )

        best_fed_state = copy.deepcopy(
            global_model.state_dict()
        )

global_model.load_state_dict(
    best_fed_state
)

print(
    "Best FedAvg validation:",
    f"{best_fed_accuracy*100:.2f}%"
)"""),

    ("markdown", "## Cell 29 — FedAvg test"),
    ("code", """fedavg_test = evaluate_tabular_model(
    global_model,
    test_loader
)

print(
    "FEDAVG MLP TEST"
)

print(
    "Accuracy:",
    f"{fedavg_test['accuracy']*100:.2f}%"
)

print(
    "Balanced Accuracy:",
    f"{fedavg_test['balanced_accuracy']*100:.2f}%"
)

print(
    "Macro F1:",
    f"{fedavg_test['macro_f1']*100:.2f}%"
)

print(
    "AUC:",
    f"{fedavg_test['auc']:.4f}"
)"""),

    ("markdown", "# STAGE 4 — DP-FedAvg MLP\nFor your feature dataset, DP is much cheaper than for ResNet50.\n\n## Cell 30 — DP configuration"),
    ("code", """DP_ROUNDS = 50
DP_LOCAL_EPOCHS = 3

CLIP_NORM = 1.0

DELTA = 1e-6

NOISE_LEVELS = [
    0.0,
    3.5,
    8.0
]

print(
    "DP noise levels:",
    NOISE_LEVELS
)

print(
    "Clip norm:",
    CLIP_NORM
)

print(
    "Delta:",
    DELTA
)"""),

    ("markdown", "## Cell 31 — DP client update"),
    ("code", """def train_dp_client(
    global_state,
    loader,
    noise_multiplier,
    clip_norm
):

    model = MessidorMLP(
        INPUT_SIZE
    ).to(DEVICE)

    model.load_state_dict(
        copy.deepcopy(
            global_state
        )
    )

    optimizer = optim.SGD(
        model.parameters(),
        lr=1e-3,
        momentum=0.9
    )

    criterion = nn.CrossEntropyLoss(
        reduction="none"
    )

    model.train()

    for epoch in range(
        DP_LOCAL_EPOCHS
    ):

        for X, y in loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            batch_size = y.size(0)

            accumulated_grads = [
                torch.zeros_like(
                    parameter
                )
                for parameter in model.parameters()
            ]

            for sample_index in range(
                batch_size
            ):

                optimizer.zero_grad()

                sample_X = X[
                    sample_index:
                    sample_index + 1
                ]

                sample_y = y[
                    sample_index:
                    sample_index + 1
                ]

                outputs = model(
                    sample_X
                )

                loss = criterion(
                    outputs,
                    sample_y
                ).mean()

                loss.backward()

                total_norm = torch.sqrt(
                    sum(
                        parameter.grad
                        .detach()
                        .pow(2)
                        .sum()
                        for parameter
                        in model.parameters()
                        if parameter.grad
                        is not None
                    )
                )

                clipping_factor = min(
                    1.0,
                    clip_norm /
                    (
                        total_norm.item()
                        + 1e-12
                    )
                )

                for index, parameter in enumerate(
                    model.parameters()
                ):

                    if parameter.grad is not None:

                        accumulated_grads[
                            index
                        ] += (
                            parameter.grad.detach()
                            *
                            clipping_factor
                        )

            for index, parameter in enumerate(
                model.parameters()
            ):

                accumulated_grads[index] /= (
                    batch_size
                )

                if noise_multiplier > 0:

                    noise = torch.randn_like(
                        accumulated_grads[index]
                    )

                    noise *= (
                        noise_multiplier
                        *
                        clip_norm
                        /
                        batch_size
                    )

                    accumulated_grads[index] += (
                        noise
                    )

                parameter.grad = (
                    accumulated_grads[index]
                )

            optimizer.step()

    return (
        model.state_dict(),
        len(loader.dataset)
    )"""),

    ("markdown", "## Cell 32 — DP-FedAvg function"),
    ("code", """def run_dp_fedavg(
    noise_multiplier
):

    model = MessidorMLP(
        INPUT_SIZE
    ).to(DEVICE)

    history = []

    best_state = copy.deepcopy(
        model.state_dict()
    )

    best_accuracy = 0

    best_round = 0

    for round_number in range(
        1,
        DP_ROUNDS + 1
    ):

        client_states = []
        client_sizes = []

        for client_id in range(
            NUM_CLIENTS
        ):

            state, size = (
                train_dp_client(
                    model.state_dict(),
                    client_loaders[
                        client_id
                    ],
                    noise_multiplier,
                    CLIP_NORM
                )
            )

            client_states.append(
                state
            )

            client_sizes.append(
                size
            )

        aggregated_state = fedavg(
            client_states,
            client_sizes
        )

        model.load_state_dict(
            aggregated_state
        )

        val_metrics = (
            evaluate_tabular_model(
                model,
                val_loader
            )
        )

        history.append({

            "round":
                round_number,

            "accuracy":
                val_metrics[
                    "accuracy"
                ],

            "macro_f1":
                val_metrics[
                    "macro_f1"
                ],

            "auc":
                val_metrics[
                    "auc"
                ]
        })

        print(
            f"Noise {noise_multiplier} | "
            f"Round {round_number:03d} | "
            f"Accuracy "
            f"{val_metrics['accuracy']*100:.2f}%"
        )

        if (
            val_metrics["accuracy"]
            > best_accuracy
        ):

            best_accuracy = (
                val_metrics["accuracy"]
            )

            best_round = (
                round_number
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

    model.load_state_dict(
        best_state
    )

    return (
        model,
        pd.DataFrame(history),
        best_accuracy,
        best_round
    )"""),

    ("markdown", "# Cell 33 — DP-FedAvg σ = 0\nThis is your privacy-free control."),
    ("code", """dp_model_0, history_0, acc_0, round_0 = (
    run_dp_fedavg(
        noise_multiplier=0.0
    )
)

print(
    "Best validation accuracy:",
    f"{acc_0*100:.2f}%"
)"""),

    ("markdown", "# Cell 34 — DP-FedAvg σ = 3.5"),
    ("code", """dp_model_35, history_35, acc_35, round_35 = (
    run_dp_fedavg(
        noise_multiplier=3.5
    )
)

print(
    "Best validation accuracy:",
    f"{acc_35*100:.2f}%"
)"""),

    ("markdown", "# Cell 35 — DP-FedAvg σ = 8"),
    ("code", """dp_model_8, history_8, acc_8, round_8 = (
    run_dp_fedavg(
        noise_multiplier=8.0
    )
)

print(
    "Best validation accuracy:",
    f"{acc_8*100:.2f}%"
)"""),

    ("markdown", "# Cell 36 — Final test evaluation"),
    ("code", """dp0_test = evaluate_tabular_model(
    dp_model_0,
    test_loader
)

dp35_test = evaluate_tabular_model(
    dp_model_35,
    test_loader
)

dp8_test = evaluate_tabular_model(
    dp_model_8,
    test_loader
)

print(
    "DP-FedAvg σ=0:",
    f"{dp0_test['accuracy']*100:.2f}%"
)

print(
    "DP-FedAvg σ=3.5:",
    f"{dp35_test['accuracy']*100:.2f}%"
)

print(
    "DP-FedAvg σ=8:",
    f"{dp8_test['accuracy']*100:.2f}%"
)"""),

    ("markdown", "# Cell 37 — Final four-stage table"),
    ("code", """final_results = pd.DataFrame([

    {
        "Stage":
            "Stage 2",

        "Method":
            "Local MLP",

        "Accuracy":
            local_test["accuracy"],

        "Balanced Accuracy":
            local_test[
                "balanced_accuracy"
            ],

        "Macro F1":
            local_test[
                "macro_f1"
            ],

        "AUC":
            local_test["auc"]
    },

    {
        "Stage":
            "Stage 3",

        "Method":
            "FedAvg MLP",

        "Accuracy":
            fedavg_test["accuracy"],

        "Balanced Accuracy":
            fedavg_test[
                "balanced_accuracy"
            ],

        "Macro F1":
            fedavg_test[
                "macro_f1"
            ],

        "AUC":
            fedavg_test["auc"]
    },

    {
        "Stage":
            "Stage 4",

        "Method":
            "DP-FedAvg σ=3.5",

        "Accuracy":
            dp35_test["accuracy"],

        "Balanced Accuracy":
            dp35_test[
                "balanced_accuracy"
            ],

        "Macro F1":
            dp35_test[
                "macro_f1"
            ],

        "AUC":
            dp35_test["auc"]
    },

    {
        "Stage":
            "Stage 4",

        "Method":
            "DP-FedAvg σ=8.0",

        "Accuracy":
            dp8_test["accuracy"],

        "Balanced Accuracy":
            dp8_test[
                "balanced_accuracy"
            ],

        "Macro F1":
            dp8_test[
                "macro_f1"
            ],

        "AUC":
            dp8_test["auc"]
    }

])

display(
    final_results.style.format({

        "Accuracy":
            "{:.2%}",

        "Balanced Accuracy":
            "{:.2%}",

        "Macro F1":
            "{:.2%}",

        "AUC":
            "{:.4f}"
    })
)"""),

    ("markdown", "# Cell 38 — Confusion matrices"),
    ("code", """fig, axes = plt.subplots(
    1,
    4,
    figsize=(20, 5)
)

experiments = [
    (
        "Local",
        local_test
    ),
    (
        "FedAvg",
        fedavg_test
    ),
    (
        "DP-FedAvg σ=3.5",
        dp35_test
    ),
    (
        "DP-FedAvg σ=8",
        dp8_test
    )
]

for ax, (name, result) in zip(
    axes,
    experiments
):

    cm = confusion_matrix(
        result["labels"],
        result["predictions"]
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()

plt.show()"""),

    ("markdown", "# Cell 39 — FedAvg convergence"),
    ("code", """fed_history_df = pd.DataFrame(
    fed_history
)

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    fed_history_df["round"],
    fed_history_df["accuracy"] * 100
)

plt.xlabel(
    "Federated Round"
)

plt.ylabel(
    "Validation Accuracy (%)"
)

plt.title(
    "Messidor FedAvg Convergence"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.show()"""),

    ("markdown", "# Cell 40 — DP comparison"),
    ("code", """plt.figure(
    figsize=(10, 6)
)

plt.plot(
    history_35["round"],
    history_35["accuracy"] * 100,
    label="DP σ=3.5"
)

plt.plot(
    history_8["round"],
    history_8["accuracy"] * 100,
    label="DP σ=8"
)

plt.xlabel(
    "Federated Round"
)

plt.ylabel(
    "Validation Accuracy (%)"
)

plt.title(
    "Messidor DP-FedAvg Convergence"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.show()"""),

    ("markdown", "# Cell 41 — Save final results"),
    ("code", """RESULT_DIR = (
    MESSIDOR_OUTPUT /
    "results"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

final_results.to_csv(
    RESULT_DIR /
    "messidor_four_stage_results.csv",
    index=False
)

pd.DataFrame(
    fed_history
).to_csv(
    RESULT_DIR /
    "fedavg_history.csv",
    index=False
)

history_35.to_csv(
    RESULT_DIR /
    "dp_fedavg_sigma_3_5_history.csv",
    index=False
)

history_8.to_csv(
    RESULT_DIR /
    "dp_fedavg_sigma_8_history.csv",
    index=False
)

print(
    "Results saved to:",
    RESULT_DIR
)"""),

    ("markdown", """# Final Notes
The completed experiment will be:
```text
MESSIDOR FEATURES
1151 samples
19 features
2 classes

              EDA
               |
               v
       Train/Val/Test
           70/15/15
               |
               v
       StandardScaler
               |
       ┌───────┴────────┐
       │                │
       v                v
   LOCAL MLP          5 CLIENTS
       │                │
       │             FedAvg
       │                │
       │                v
       │          GLOBAL MLP
       │                │
       └───────┬────────┘
               │
               v
          DP-FedAvg
          σ = 3.5
          σ = 8.0
               |
               v
     Final Test Comparison
```

### One important research distinction
Your Messidor CSV is a binary classification feature dataset, so its results should not be directly merged with your IDRiD five-class image results.
Your paper/project can instead contain two experiments:
*   IDRiD (Fundus images) -> 5-class DR -> ResNet50
*   Messidor CSV (19 extracted features) -> Binary -> MLP
For both: Local -> FedAvg -> DP-FedAvg. This is much cleaner scientifically than trying to feed Messidor features into ResNet50.

For the DP stage, the next improvement recommended is replacing the explicit per-sample loop with Opacus + a formal RDP privacy accountant, because that will let you report a defensible ε, δ alongside accuracy rather than only reporting the noise multiplier.
""")
]

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

for ctype, csource in cells_data:
    cell = {
        "cell_type": ctype,
        "metadata": {},
        "source": [line + '\\n' for line in csource.split('\\n')]
    }
    # Clean up the last newline to match standard jupyter format
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip('\\n')
        
    if ctype == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    nb["cells"].append(cell)

with open(r"D:\RMS\Messidor_Four_Stage_Pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook generated successfully!")
