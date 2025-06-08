from implicit.als import AlternatingLeastSquares
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

data_path = "..."

MEMBERS = f"{data_path}/members.csv"
SONG_FEAT = f"{data_path}/song_extra_info.csv"
SONGS = f"{data_path}/songs.csv"
TRAIN = f"{data_path}/train.csv"
MERGE = f"{data_path}/train_data.csv"

members_df = pd.read_csv(MEMBERS)
song_feat_df = pd.read_csv(SONG_FEAT)
songs_df = pd.read_csv(SONGS)
train_df = pd.read_csv(TRAIN)

train_df = train_df.groupby("msno").sample(frac=0.1, random_state=42)

le_msno = LabelEncoder()
le_song_id = LabelEncoder()

le_msno.fit(train_df["msno"])
le_song_id.fit(train_df["song_id"])

train = pd.DataFrame({"msno": le_msno.transform(train_df["msno"].values), "song_id": le_song_id.transform(train_df["song_id"].values), "target": train_df["target"]})

private_data = train.groupby("msno").sample(frac=0.3, random_state=42)
global_data = train.drop(private_data.index).sort_values(by="msno")

private_data.shape, global_data.shape

# --- Конфигурация федерации ---
NUM_CLIENTS = 5
LOCAL_EPOCHS = 3
AGGREGATION_ROUNDS = 10
FACTORS = 50
REGULARIZATION = 0.01
TOP_N_RECOMMENDATIONS = 20  # число рекомендаций для генерации новых данных

# 1. Инициализация глобальной модели
global_model = AlternatingLeastSquares(factors=FACTORS,
                                      regularization=REGULARIZATION,
                                      use_gpu=False)

user_item = csr_matrix(
            (global_data['target'],
            (global_data['msno'], global_data['song_id']))
        )

global_model.fit(user_item)

train_data, test_data = train_test_split(train, test_size=0.2, random_state=42, shuffle=True)

test_users = test_data["msno"]
test_items = test_data["song_id"]

user_factors = global_model.user_factors[test_users]
song_factors = global_model.item_factors[test_items]
predicted_scores = np.sum(user_factors * song_factors, axis=1)

auc = roc_auc_score(test_data["target"], predicted_scores)
precision = precision_score(test_data['target'], predicted_scores>0.6)
recall = recall_score(test_data['target'], predicted_scores>0.6)
print("AUC: ", auc)
print("precision: ", precision)
print("recall: ", recall)

# with open('als_federated_model.pkl', 'rb') as f:
#     global_model = pickle.load(f)

class Client:
    def __init__(self, client_id, base_data):
        self.id = client_id
        self.base_data = base_data
        self.model = AlternatingLeastSquares(factors=FACTORS,
                                            regularization=REGULARIZATION,
                                            use_gpu=False, iterations=5)

    def local_update(self, global_user_factors, global_item_factors):
        self.model.user_factors = global_user_factors.copy()
        self.model.item_factors = global_item_factors.copy()

        user_items = csr_matrix(
            (self.base_data['target'],
             (np.zeros(len(self.base_data)), 
              self.base_data['song_id'])),
            shape=(1, global_item_factors.shape[0])
        )

        self.model.partial_fit_users([self.id], user_items)

        return self.model.user_factors, self.model.item_factors

private_data[private_data["msno"] == private_data["msno"].unique()[0]]

clients = []
for cid in tqdm(private_data["msno"].unique()):
    base_data = (private_data[private_data["msno"] == cid]).copy()
    clients.append(Client(cid, base_data))

def aggregate(models_user, models_item):
    avg_user = np.mean(models_user, axis=0)
    avg_item = np.mean(models_item, axis=0)
    return avg_user, avg_item

for round_idx in tqdm(range(AGGREGATION_ROUNDS)):
    collected_user = []
    collected_item = []
    for client in clients[:10]:
        u_f, i_f = client.local_update(global_model.user_factors, global_model.item_factors)
        collected_user.append(u_f)
        collected_item.append(i_f)
    mean_user, mean_item = aggregate(collected_user, collected_item)
    global_model.user_factors = mean_user
    global_model.item_factors = mean_item
    print(f"Round {round_idx+1}/{AGGREGATION_ROUNDS} aggregated with synthetic data")

test_users = test_data["msno"]
test_items = test_data["song_id"]

user_factors = global_model.user_factors[test_users]
song_factors = global_model.item_factors[test_items]
predicted_scores = np.sum(user_factors * song_factors, axis=1)

auc = roc_auc_score(test_data["target"], predicted_scores)
precision = precision_score(test_data['target'], predicted_scores>0.6)
recall = recall_score(test_data['target'], predicted_scores>0.6)
print("AUC: ", auc)
print("precision: ", precision)
print("recall: ", recall)

with open('als_federated_c_model.pkl', 'wb') as f:
    pickle.dump(global_model, f)