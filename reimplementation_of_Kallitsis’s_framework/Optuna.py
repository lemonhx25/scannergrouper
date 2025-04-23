import glob
import os
from collections import Counter

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

os.environ["OMP_NUM_THREADS"] = '1'

import random
import time
from datetime import datetime

import optuna

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, accuracy_score, jaccard_score, classification_report,confusion_matrix
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomDataset(Dataset):
    def __init__(self, features):
        """

        """
        self.features = torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        """

        """
        super(Autoencoder, self).__init__()

        # 
        # 
        encoder_layers = []
        # 
        previous_dim = input_dim
        # 
        for hidden_dim in hidden_dims:
            # 
            encoder_layers.append(nn.Linear(previous_dim, hidden_dim))
            # 
            encoder_layers.append(nn.ReLU())
            # 
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            # 
            previous_dim = hidden_dim
        # 
        encoder_layers.append(nn.Linear(previous_dim, latent_dim))
        # 
        encoder_layers.append(nn.ReLU())
        # 
        self.encoder = nn.Sequential(*encoder_layers)

        # 
        # 
        decoder_layers = []
        # 
        previous_dim = latent_dim
        # 
        for hidden_dim in reversed(hidden_dims):
            # 
            decoder_layers.append(nn.Linear(previous_dim, hidden_dim))
            # 
            decoder_layers.append(nn.ReLU())
            # 
            decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            # 
            previous_dim = hidden_dim
        # 
        decoder_layers.append(nn.Linear(previous_dim, input_dim))
        # 
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x, latent=False):
        """

        """
        z = self.encoder(x)  # 
        if latent:
            return z  # 
        x_reconstructed = self.decoder(z)  # 
        return x_reconstructed  # 


class RMSELoss(nn.Module):
    def __init__(self):
        """
        """
        super(RMSELoss, self).__init__()

    def forward(self, outputs, targets):
        """

        """
        # 
        mse_loss = ((outputs - targets) ** 2).sum()

        # 
        norm_factor = (targets ** 2).sum()

        # 
        return mse_loss / norm_factor


def wasserstein_distance(u_values, v_values):
    """

    """
    # 
    cost_matrix = pairwise_distances(u_values, v_values, metric='euclidean')

    # 
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 
    distance = cost_matrix[row_ind, col_ind].sum()

    return distance


def load_data(directory):
    """

    """
    # 
    all_files = glob.glob(os.path.join(directory, 'processed_*.csv'))

    # 
    list_df = []

    # 
    for file in all_files:
        # 
        base_name = os.path.basename(file)
        # 
        parts = base_name.split('_')
        date_str = parts[1] + parts[2]  # 
        # 
        df = pd.read_csv(file)
        # 
        df['Date'] = date_str
        # 
        list_df.append(df)

    # 
    return pd.concat(list_df, ignore_index=True)


# 
def preprocess_data(df):
    # 
    #df = df.loc[df['Total Packets'] >= 3].copy()
    iplist=list(df['Scanner IP'])
    # 
    df.drop(columns=['Scanner IP', 'Session First Packet Nos', 'Set of TTL Values', 'Network Prefix'], errors='ignore',
            inplace=True)

    # 
    df.loc[:, 'TCP Options'] = df['TCP Options'].astype(int)

    # 
    ports = df['Set of Ports Scanned'].apply(lambda x: eval(x))
    all_ports = sorted(set.union(*ports))
    for port in all_ports:
        df.loc[:, f'Port_{port}'] = ports.apply(lambda x: 1 if port in x else 0)
    df.drop('Set of Ports Scanned', axis=1, inplace=True)

    # 
    # df[[f'Port_{port}' for port in all_ports]].to_csv('ports_one_hot_encoded.csv', index=False)

    # 
    protocols = df['Protocol Types'].apply(lambda x: eval(x))
    all_protocols = sorted(set.union(*protocols))
    for protocol in all_protocols:
        df.loc[:, f'Protocol_{protocol}'] = protocols.apply(lambda x: 1 if protocol in x else 0)
    df.drop('Protocol Types', axis=1, inplace=True)

    # 
    df[[f'Protocol_{protocol}' for protocol in all_protocols]].to_csv('protocols_one_hot_encoded.csv', index=False)

    # 
    # 
    df = pd.get_dummies(df, columns=['Device Type', 'Destination Strategy', 'IPID Strategy'])

    # 
    features_to_scale = ['Total Packets', 'Total Bytes', 'Average Inter-arrival Time',
                         'Number of Distinct Destination Ports', 'Number of Distinct Destination Addresses',
                         'Prefix Density']
    # 
    features_to_scale += [f'Port_{port}' for port in all_ports]
    features_to_scale += [f'Protocol_{protocol}' for protocol in all_protocols]

    # 
    scaler = StandardScaler()  # 
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    # 
    total_features_needed = 25
    

    for item in ['Protocol_ICMP Time Exceeded']:
        if item not in df.columns.tolist():
            df[f'{item}'] = 0 
            
    current_feature_count = df.shape[1]
    
    if current_feature_count < total_features_needed:
        for i in range(total_features_needed - current_feature_count):
            df[f'Extra_Feature_{i + 1}'] = 0  #


    for column in df.select_dtypes(include=['bool']).columns:
        df[column] = df[column].astype(int)

    # 
    print("Processed data preview (head):")
    print(df.head())

    # 
    print("Processed data statistics (describe):")
    print(df.describe())

    # 
    return df,iplist


def elbow_method(latent_representation, max_clusters=50):
    """

    """
    # 
    sse = []
    # 
    for k in range(1, max_clusters + 1):
        # 
        kmeans = KMeans(n_clusters=k, n_init=50, random_state=42)
        # 
        kmeans.fit(latent_representation)
        # 
        sse.append(kmeans.inertia_)

    # 
    plt.figure()
    # 
    plt.plot(range(1, max_clusters + 1), sse, marker='o')
    # 
    plt.xlabel('Number of clusters')
    # 
    plt.ylabel('Sum of squared distances')
    # 
    plt.title('Elbow Method For Optimal k')
    # 
    plt.show()

    # 
    optimal_k = np.argmax(np.diff(sse)) + 2  # 
    return optimal_k


def perform_clustering(latent_representation, n_clusters):
    """
 
    """
    # 
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=42)
    # 
    kmeans.fit(latent_representation)
    # 
    labels = kmeans.labels_
    # 
    centroids = kmeans.cluster_centers_
    # 
    return labels, centroids




def visualize_clustering(latent_representation, labels, centroids, save_path=None):
    """

    """
    # 
    plt.figure()
    plt.scatter(latent_representation[:, 0], latent_representation[:, 1], c=labels, s=50, cmap='viridis')
    # 
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='X')
    # 
    plt.title('K-means Clustering')
    plt.xlabel('Latent Feature 1')
    plt.ylabel('Latent Feature 2')
    # 
    if save_path:
        plt.savefig(save_path)
    # 
    plt.show()



def detect_change_points(centroids_by_day, threshold=1.5, p_value=0.036):
    """

    """
    # 
    change_scores = []
    # 
    dates = list(centroids_by_day.keys())

    # 
    for i in range(1, len(dates)):
        # 
        centroids_1 = centroids_by_day[dates[i - 1]]
        centroids_2 = centroids_by_day[dates[i]]
        # 
        change_score = wasserstein_distance(centroids_1, centroids_2)
        change_scores.append(change_score)

    # 
    plt.plot(range(1, len(dates)), change_scores, marker='o')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold with p-val={p_value}')
    plt.xlabel('Date Index')
    plt.ylabel('Wasserstein Distance b/w adjacent days')
    plt.title('Wasserstein Distance Over Time')
    plt.legend()
    plt.show()


def train_autoencoder(dataloader, input_dim, hidden_dims, latent_dim, num_epochs, learning_rate, regularization_weight,
                      log_filename, result_dir):
    """

    """
    # 
    model = Autoencoder(input_dim, hidden_dims, latent_dim).to(device)

    # 
    criterion = RMSELoss()
    # 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization_weight)

    # 
    model.train()
    # 
    losses = []
    best_loss = float('inf')  # 

    # 
    for epoch in range(num_epochs):
        epoch_losses = []  # 
        epoch_start_time = time.time()  # 

        # 
        for batch_x in dataloader:
            # 
            batch_x = batch_x.to(device)
            # 
            optimizer.zero_grad()
            # 
            outputs = model(batch_x)
            # 
            loss = criterion(outputs, batch_x)
            # 
            loss.backward()
            # 
            optimizer.step()
            # 
            epoch_losses.append(loss.item())

        # 
        epoch_loss = np.mean(epoch_losses)
        losses.append(epoch_loss)

        # 
        if epoch_loss < best_loss:
            best_loss = epoch_loss

        # 
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        # 
        with open(os.path.join(result_dir, log_filename), "a") as log_file:
            log_file.write(f'Epoch {epoch + 1}, Loss: {epoch_loss}, Duration: {epoch_duration:.2f} seconds\n')
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss}, Duration: {epoch_duration:.2f} seconds')

    # 
    torch.save(model.state_dict(),
               os.path.join(result_dir, f"best_autoencoder_model_final_loss_{best_loss:.4f}.pth"))

    # 
    plt.figure()
    # 
    plt.plot(range(1, len(losses) + 1), losses, label="Training Loss")
    # 
    plt.xlabel("Epoch")
    # 
    plt.ylabel("Loss")
    # 
    plt.title("Training Loss over Epochs")
    # 
    plt.legend()
    # 
    plt.grid(True)
    # 
    plt.savefig(os.path.join(result_dir, "loss_plot.png"))
    # 
    plt.close()

    # 
    return model, np.mean(losses)
def grid_search(dataset, input_dim, hidden_dims_list, latent_dim, epochs_list, batch_sizes_list, learning_rates_list,
                regularization_weight):
    """

    """
    # 
    best_loss = float('inf')
    best_params = {}

    current_time = datetime.now().strftime('%Y%m%d_%H%M')  # 

    # 
    for num_epochs in epochs_list:
        # 
        for batch_size in batch_sizes_list:
            for hidden_dims in hidden_dims_list:
                # 
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

                for learning_rate in learning_rates_list:
                    # 
                    result_dir = f"results/{current_time}/epochs{num_epochs}_batch{batch_size}_lr{learning_rate}_hidden{hidden_dims}_latent{latent_dim}"
                    os.makedirs(result_dir, exist_ok=True)

                    # 
                    log_filename = "training_log.txt"
                    # 
                    print(
                        f"Testing parameters: epochs={num_epochs}, batch_size={batch_size}, learning_rate={learning_rate}, "
                        f"regularization_weight={regularization_weight}, input_dim={input_dim}, hidden_dims={hidden_dims}, "
                        f"latent_dim={latent_dim}")

                    # 
                    with open(os.path.join(result_dir, log_filename), "a") as log_file:
                        log_file.write(
                            f"Testing parameters: epochs={num_epochs}, batch_size={batch_size}, learning_rate={learning_rate}, "
                            f"regularization_weight={regularization_weight}, input_dim={input_dim}, hidden_dims={hidden_dims}, "
                            f"latent_dim={latent_dim}\n")

                    # 
                    _, loss = train_autoencoder(dataloader, input_dim, hidden_dims, latent_dim, num_epochs,
                                                learning_rate,
                                                regularization_weight, log_filename, result_dir)
                    # 
                    if loss < best_loss:
                        best_loss = loss
                        best_params = {
                            'epochs': num_epochs,
                            'batch_size': batch_size,
                            'learning_rate': learning_rate,
                            'hidden_dims': hidden_dims
                        }

    # 
    return best_params
def optimize_autoencoder_hyperparameters(dataset, input_dim, n_trials=20):
    """

    """

    # 
    main_current_time = datetime.now().strftime('%Y%m%d_%H%M')
    main_result_dir = f"results/{main_current_time}/"
    os.makedirs(main_result_dir, exist_ok=True)

    # 
    study = optuna.create_study(direction='minimize')

    # 
    def objective(trial):
        # 
        num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 2)

        # 
        hidden_dims = []
        for i in range(num_hidden_layers):
            hidden_dim = trial.suggest_int(f'hidden_dim_{i + 1}', 100, 1000, step=100)
            hidden_dims.append(hidden_dim)

        # 
        latent_dim = trial.suggest_int('latent_dim', 10, 50)
        # 
        num_epochs = trial.suggest_categorical('epochs', [50, 100, 200])
        # 
        batch_size = trial.suggest_categorical('batch_size', [2048, 1024, 512, 256, 128, 64, 32, 16])
        # 
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        # 
        regularization_weight = 0.001

        # 
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        # 
        current_time = datetime.now().strftime('%Y%m%d_%H%M')
        # 
        param_str = (f"{current_time}_epochs{num_epochs}_batch{batch_size}_lr{learning_rate:.4f}_"
                     f"hidden{hidden_dims}_latent{latent_dim}_reg{regularization_weight:.4f}")
        result_dir = os.path.join(main_result_dir, param_str)
        os.makedirs(result_dir, exist_ok=True)
        log_filename = "training_log.txt"


        # 
        print(f"Trial {trial.number}: hidden_dims={hidden_dims}, latent_dim={latent_dim}, "
              f"num_epochs={num_epochs}, batch_size={batch_size}, learning_rate={learning_rate}, "
              f"regularization_weight={regularization_weight}")

        # 
        _, loss = train_autoencoder(dataloader, input_dim, hidden_dims, latent_dim, num_epochs, learning_rate,
                                    regularization_weight, log_filename, result_dir)

        return loss

    # 
    study.optimize(objective, n_trials=n_trials)

    # 
    print("Best trial:")
    trial = study.best_trial

    print("  Loss: {}".format(trial.value))
    print("  Best hyperparameters: {}".format(trial.params))

    return trial
def train_autoencoder_with_fixed_params(dataset, input_dim):
    """

    """
    # 
    hidden_dims = [1000]  # 
    latent_dim = 50  # 
    num_epochs = 50#50 # 
    batch_size = 2000  # 
    learning_rate = 0.0003  # 
    regularization_weight = 0.001  # 

    # 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 
    current_time = datetime.now().strftime('%Y%m%d_%H%M')
    result_dir = f"results/{current_time}_fixed_params"  # 
    os.makedirs(result_dir, exist_ok=True)  # 
    log_filename = "training_log.txt"  # 

    # 
    with open(os.path.join(result_dir, log_filename), "a") as log_file:
        log_file.write(f"Training Parameters: hidden_dims={hidden_dims}, latent_dim={latent_dim}, "
                       f"num_epochs={num_epochs}, batch_size={batch_size}, learning_rate={learning_rate}, "
                       f"regularization_weight={regularization_weight}\n")

    # 
    model, loss = train_autoencoder(dataloader, input_dim, hidden_dims, latent_dim, num_epochs, learning_rate,
                                    regularization_weight, log_filename, result_dir)
    torch.save(model.state_dict(),
               os.path.join(result_dir, f"best_autoencoder_model_final_loss_{loss:.4f}.pth"))
    # 
    print(f"Final Loss: {loss}")

    # 
    return model, loss

# 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_and_preprocess_data(file_path,dataset_name,ip_filter=0):
    """

    """
    # # 
    # all_data = load_data(data_directory)
    # # 
    # features_scaled = preprocess_data(all_data)
    # 
    df = pd.read_csv(file_path)
    print(df)
    print(f"Original number of rows: {df.shape[0]}")  # 
    import pickle
    
    if (ip_filter==1):
        # 
        if(dataset_name=='SelfDeploy24'):
            file_path_iplist = 'iplist_selfdeploy24_test.pkl'
        if(dataset_name=='SelfDeploy25'):
            file_path_iplist = 'iplist_selfdeploy25_test.pkl'
        
        # 
        with open(file_path_iplist, 'rb') as file:
            loaded_list = pickle.load(file)
        
        #print(f"Loaded list: {loaded_list}")
        df = df[df['Scanner IP'].isin(loaded_list)]
        print(f"Filtered number of rows: {df.shape[0]}")  # 

    # 
    #df = filter_small_classes(df, label_column='Label', min_samples=5)
    
    
    # 
    features_scaled,iplist = preprocess_data(df)
    # 
    #features_scaled.to_csv('Processed_Features_Scaled1.csv', index=False)
    #features_scaled.to_csv(file_path.split('/')[-1]+'Processed_Features_Scaled1.csv', index=False)
    # # 
    # features_for_tensor = features_scaled.drop(columns=['Date'], errors='ignore')
    # 
    features_for_tensor = features_scaled.drop(columns=['Label'], errors='ignore')
    # 
    features_array = features_for_tensor.to_numpy().astype(np.float32)
    # 
    dataset = CustomDataset(features_array)
    # 
    input_dim = features_array.shape[1]
    return features_scaled, input_dim, dataset,iplist


def process_and_cluster_daily_data(data_directory, model_path, device):
    """

    """
    # 
    features_scaled, input_dim, dataset = load_and_preprocess_data(data_directory)

    # 
    unique_dates = features_scaled['Date'].unique()
    print(unique_dates)

    # 
    input_dim = features_scaled.shape[1] - 1
    hidden_dims = [1000]
    latent_dim = 50

    model = Autoencoder(input_dim, hidden_dims, latent_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 
    daily_data = {}
    centroids_by_day = {}

    for date in unique_dates:
        # 
        daily_data[date] = features_scaled[features_scaled['Date'] == date].drop(columns=['Date'], errors='ignore')

    # 
    for date, data in daily_data.items():
        features_array = data.to_numpy().astype(np.float32)

        print(f"Original data dimensions for {date}: {features_array.shape}")

        # 
        latent_representation = model.encoder(torch.tensor(features_array).to(device)).detach().cpu().numpy()

        print(f"Encoded data dimensions for {date}: {latent_representation.shape}")

        # 
        optimal_k = elbow_method(latent_representation)
        print(f"Optimal number of clusters for {date}: {optimal_k}")

        # 
        labels, centroids = perform_clustering(latent_representation, n_clusters=optimal_k)
        centroids_by_day[date] = centroids

        # 
        result_dir = f"results/visualizations/clustering_{date}_k{optimal_k}"
        os.makedirs(result_dir, exist_ok=True)
        np.savetxt(os.path.join(result_dir, f'labels_{date}.csv'), labels, delimiter=",")
        np.savetxt(os.path.join(result_dir, f'centroids_{date}.csv'), centroids, delimiter=",")

        # 
        visualize_clustering(latent_representation, labels, centroids, save_path=os.path.join(result_dir, 'clustering.png'))

        print(f"Clustering for {date} completed.")

    # 
    detect_change_points(centroids_by_day)

    return centroids_by_day

def filter_small_classes(df, label_column, min_samples=5):
    """

    """
    # 
    label_counts = df[label_column].value_counts()

    # 
    valid_labels = label_counts[label_counts >= min_samples].index

    # 
    filtered_df = df[df[label_column].isin(valid_labels)].copy()

    return filtered_df


def semi_supervised_evaluation(latent_representation_train, labels_train,latent_representation_test, labels_test,iplist_test,dataset_name):
    """
    """
    # 
    # X_train, X_test, y_train, y_test = train_test_split(latent_representation, labels, test_size=0.2, random_state=42)

    # 
    #train_x, test_x, train_y, test_y = train_test_split(latent_representation, labels, test_size=0.2, random_state=32,stratify=labels)

    train_x=latent_representation_train
    test_x=latent_representation_test
    train_y=labels_train
    test_y=labels_test

    
    # 
    train_distribution = Counter(train_y)
    test_distribution = Counter(test_y)
    #original_distribution = Counter(labels)

    print("Train Label distribution:")
    print(train_distribution)

    print("Test Label distribution:")
    print(test_distribution)

    index_censys=test_y=='censys'
    print(index_censys)
    index_driftnet=test_y=='driftnet'
    index_unknown=test_y=='unknown'
    
    test_y[index_censys] = 'unknown'
    test_y[index_driftnet] = 'unknown'


    index_censys_train=train_y=='censys'
    index_driftnet_train=train_y=='driftnet'
    index_unknown_train=train_y=='unknown'
    
    train_y[index_censys_train] = 'unknown'
    train_y[index_driftnet_train] = 'unknown'
    

    #print("Original Label distribution:")
    #print(original_distribution)
    
    # 
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(train_x, train_y)

    
    

    # 
    y_pred = knn.predict(test_x)

    y_pred = np.delete(y_pred, list(index_unknown))
    test_y = np.delete(test_y, list(index_unknown))
    iplist_test = np.delete(iplist_test, list(index_unknown))
    
    
    test_y[test_y=='unknown']='censys+driftnet'
    y_pred[y_pred=='unknown']='censys+driftnet'

    final_results_ytrue = list(vote_for_ip_addresses(iplist_test, test_y).values())
    final_results_ypred = list(vote_for_ip_addresses(iplist_test, y_pred).values())

    # 
    #report = classification_report(test_y, y_pred, digits=2, zero_division=1)
    report = classification_report(final_results_ytrue, final_results_ypred, digits=2, zero_division=np.nan)

    #from sklearn.metrics import classification_report

    def save_classification_report(y_true,y_pred,save_path):
        
        acc_report_df = pd.DataFrame(classification_report(y_true, y_pred,output_dict=True,zero_division=np.nan)).T
        acc_report_df.iloc[-3,:2]= np.nan
        acc_report_df.iloc[-3,3]= acc_report_df.iloc[-2,3]
        acc_report_df.to_csv(save_path,float_format='%.3f')
        return acc_report_df.round(2)
    save_path='report/'+dataset_name.lower()+'.csv'
    acc_report_df = save_classification_report(final_results_ytrue, final_results_ypred,save_path=save_path)

    print(report)

    # 
    #accuracy = accuracy_score(test_y, y_pred)
    accuracy = accuracy_score(final_results_ytrue, final_results_ypred)
    print(f'Accuracy: {accuracy:.2f}')

    cm = confusion_matrix(final_results_ytrue, final_results_ypred)
    print(cm)

    return report, accuracy


from collections import defaultdict

def vote_for_ip_addresses(l, y):
    # 
    ip_dict = defaultdict(list)
    
    # 
    for ip, result in zip(l, y):
        ip_dict[ip].append(result)
    
    # 
    final_results = {}
    for ip, results in ip_dict.items():
        # 
        count = defaultdict(int)
        for result in results:
            count[result] += 1
        
        # 
        max_count = max(count.values())
        for result, cnt in count.items():
            if cnt == max_count:
                final_results[ip] = result
                break
    
    return final_results


import argparse


def main():
    """
    
    """
    parser = argparse.ArgumentParser(description='Process dataset')
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset', required=True)
    args = parser.parse_args()
    dataset_name=args.dataset_name
    
    
    seed = 42   
    set_seed(seed)   

    
    if torch.cuda.is_available():
        print("CUDA is available. Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")


    print("Using device:", device)

    # 
    # data_directory = 'dataset/processed_csv'
    # model_path = 'results/20240814_1528_fixed_params/best_autoencoder_model_final_loss_0.0025.pth'
    #
    # 
    # centroids_by_day = process_and_cluster_daily_data(data_directory, model_path, device)
    if(dataset_name=='SelfDeploy24'):
        #
        file_path = 'output/selfdeploy24_merged_dataset_with_labels_all.csv'
        file_path_train = 'output/selfdeploy24_merged_dataset_with_labels_train.csv'
        file_path_test = 'output/selfdeploy24_merged_dataset_with_labels_test.csv'
        model_path = 'results/20250423_1519_fixed_params/best_autoencoder_model_final_loss_0.0041.pth'
    if(dataset_name=='SelfDeploy25'):
        # 
        file_path = 'output/selfdeploy25_merged_dataset_with_labels_all.csv'
        file_path_train = 'output/selfdeploy25_merged_dataset_with_labels_train.csv'
        file_path_test = 'output/selfdeploy25_merged_dataset_with_labels_test.csv'
        model_path = 'results/20250423_1525_fixed_params/best_autoencoder_model_final_loss_0.0031.pth'

    
    df, input_dim, dataset,iplist = load_and_preprocess_data(file_path,dataset_name)
    df_train, input_dim_train, dataset_train,iplist_train = load_and_preprocess_data(file_path_train,dataset_name)
    df_test, input_dim_test, dataset_test,iplist_test = load_and_preprocess_data(file_path_test,dataset_name,ip_filter=1)
    print('df_test',df_test,len(iplist_test))
     
    print(f"Total number of samples in df: {len(df)}")

    
    X = df.drop(columns=['Label']).values
    y = df['Label'].values

    X_train = df_train.drop(columns=['Label']).values
    y_train = df_train['Label'].values
    
    X_test = df_test.drop(columns=['Label']).values
    y_test = df_test['Label'].values

    

    
    unique_labels, counts = np.unique(y, return_counts=True)
    label_distribution = dict(zip(unique_labels, counts))
    print("Label distribution train:")
    for label, count in label_distribution.items():
        print(f"{label}: {count} samples")

    unique_labels, counts = np.unique(y_test, return_counts=True)
    label_distribution = dict(zip(unique_labels, counts))
    print("Label distribution test:")
    for label, count in label_distribution.items():
        print(f"{label}: {count} samples")

    
    print("Loading the pre-trained autoencoder model...")
    hidden_dims = [1000]
    latent_dim = 50
    model = Autoencoder(input_dim, hidden_dims, latent_dim).to(device)

    #model, loss = train_autoencoder_with_fixed_params(dataset, input_dim)



    model.load_state_dict(torch.load(model_path))
    model.eval()

    
    print("Evaluating the clustering quality with the pre-trained model...")

    latent_representation_train = model.encoder(torch.tensor(X_train, dtype=torch.float32).to(device)).detach().cpu().numpy()

    
    
    latent_representation_test = model.encoder(torch.tensor(X_test, dtype=torch.float32).to(device)).detach().cpu().numpy()
    print('latent_representation_test',latent_representation_test)
    semi_supervised_evaluation(latent_representation_train, y_train,latent_representation_test,y_test,iplist_test,dataset_name)
    
    
    #semi_supervised_evaluation(latent_representation, y)


if __name__ == '__main__':
    main()
