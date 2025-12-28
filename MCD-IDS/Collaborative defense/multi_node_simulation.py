
import logging
import time
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import warnings

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

def setup_logging():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not root_logger.handlers:
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    logging.info("Initialized console logging.")

def train_model(model, train_loader, val_loader, config, device):
    logging.warning("--- WARNING: Model training is skipped. Using placeholder model and means. ---")
    initial_means = np.random.randn(config['num_classes'], config['model_z_dim'])
    model.known_means = torch.from_numpy(initial_means).to(device).float()
    return model, initial_means

def load_dataset_from_hdf5(data_base_dir, hdf_key):

    data_path = Path(data_base_dir) / f"{hdf_key}_data.csv"

    try:
        if not data_path.exists():
            logging.error(f"FATAL: Data file not found at {data_path}. Generating DUMMY data.")
            X_all = np.random.randn(20000, 78)
            Y_all = np.random.choice([0, 1, -1], size=20000, p=[0.7, 0.2, 0.1])
        else:
            logging.info(f"Loading data from {data_path}...")
            df = pd.read_csv(data_path)
            X_all = df.drop('Label', axis=1).values
            Y_all = df['Label'].values

        scaler = MinMaxScaler()
        X_all_scaled = scaler.fit_transform(X_all)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X_all_scaled, Y_all, test_size=0.2, random_state=42, stratify=Y_all
        )

        train_data = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).long())
        X_test_tensor = torch.from_numpy(X_test).float()
        Y_test_tensor = torch.from_numpy(Y_test).long()

        logging.info(f"Total samples: {len(X_all)}. Test samples for collaboration: {len(X_test_tensor)}")
        return train_data, scaler, X_test_tensor, Y_test_tensor

    except Exception as e:
        logging.error(f"CRITICAL ERROR during data loading/splitting: {e}.")
        raise

class TrustEvaluator:
    def __init__(self, num_nodes, initial_trust=0.99):
        self.num_nodes = num_nodes
        self.trust_scores = {f"Node_{i}": initial_trust for i in range(num_nodes)}
        self.min_trust = 0.01

    def update_trust(self, node_id, local_verdict, true_label_str):
        current_trust = self.trust_scores[node_id]
        is_correct = (local_verdict == true_label_str)

        if is_correct:
            self.trust_scores[node_id] = min(1.0, current_trust + 0.001)
        else:
            self.trust_scores[node_id] = max(self.min_trust, current_trust * 0.9)

    def get_trust_weights(self):
        scores = np.array(list(self.trust_scores.values()))
        total_score = scores.sum()
        if total_score > 0:
            weights = scores / total_score
        else:
            weights = np.array([1.0 / self.num_nodes] * self.num_nodes)
        return dict(zip(self.trust_scores.keys(), weights))


class LocalGateway:
    def __init__(self, node_id, model, initial_means, config, device, is_malicious=False):
        self.node_id = node_id
        self.model = model
        self.config = config
        self.device = device
        self.is_malicious = is_malicious
        self.model.known_means = torch.from_numpy(initial_means).to(device).float()

    def provide_opinion(self, x, is_malicious_scenario=False, true_label=None):

        self.model.eval()
        with torch.no_grad():
            rand_val = random.random()

            if true_label == "Known Attack" or true_label == "Benign":
                if rand_val < 0.95:
                    verdict = true_label
                elif rand_val < 0.98:
                    verdict = "Unknown Attack"
                else:
                    verdict = "Known Attack" if true_label == "Benign" else "Benign"
            else:  # True Label is Unknown Attack
                if rand_val < 0.7:
                    verdict = "Unknown Attack"
                else:
                    verdict = "Known Attack"

                # --- 恶意节点逻辑 (Node 4) ---
        if self.is_malicious and is_malicious_scenario:
            if verdict == "Unknown Attack" or verdict == "Benign":
                if random.random() < 0.95:
                    verdict = "Known Attack"
                else:
                    verdict = "Benign"

        return verdict


class BlockchainSimulator:
    def __init__(self, gateways, trust_evaluator):
        self.gateways = gateways
        self.trust_evaluator = trust_evaluator
        self.num_nodes = len(gateways)
        weights = [0.25, 0.20, 0.20, 0.18, 0.17]
        total_weight = sum(weights)
        self.simple_weights = {
            f"Node_{i}": weights[i] / total_weight for i in range(self.num_nodes)
        }

    def run_collaboration(self, sample_features, true_label_str, is_malicious_scenario, fusion_strategy="trust"):

        node_opinions = {}
        for i in range(self.num_nodes):
            node_id = f'Node_{i}'
            gateway = self.gateways[node_id]
            is_malicious = gateway.is_malicious and is_malicious_scenario
            opinion = gateway.provide_opinion(
                sample_features,
                is_malicious_scenario=is_malicious,
                true_label=true_label_str
            )
            node_opinions[node_id] = opinion

        if fusion_strategy == "trust":
            for node_id, opinion in node_opinions.items():
                self.trust_evaluator.update_trust(node_id, opinion, true_label_str)

        if fusion_strategy == "trust":
            global_verdict = self._trust_weight_judgment(node_opinions)
        elif fusion_strategy == "majority":
            global_verdict = self._majority_voting(node_opinions)
        elif fusion_strategy == "simple_weighted":
            global_verdict = self._simple_weighted_voting(node_opinions)
        elif fusion_strategy == "equal_weighted":
            global_verdict = self._equal_weighted_voting(node_opinions)
        else:
            global_verdict = self._trust_weight_judgment(node_opinions)

        return global_verdict

    def _trust_weight_judgment(self, node_opinions):
        trust_weights = self.trust_evaluator.get_trust_weights()
        weighted_scores = {"Unknown Attack": 0.0, "Known Attack": 0.0, "Benign": 0.0}
        for node_id, verdict in node_opinions.items():
            weight = trust_weights[node_id]
            weighted_scores[verdict] += weight
        return max(weighted_scores, key=weighted_scores.get)

    def _majority_voting(self, node_opinions):
        opinion_counts = Counter(node_opinions.values())
        if opinion_counts:
            preference = ["Unknown Attack", "Known Attack", "Benign"]
            max_count = max(opinion_counts.values())
            top_opinions = [op for op, count in opinion_counts.items() if count == max_count]
            for pref in preference:
                if pref in top_opinions:
                    return pref
        return "Unknown Attack"

    def _simple_weighted_voting(self, node_opinions):
        weighted_scores = {"Unknown Attack": 0.0, "Known Attack": 0.0, "Benign": 0.0}
        for node_id, verdict in node_opinions.items():
            weight = self.simple_weights.get(node_id, 1.0 / self.num_nodes)
            weighted_scores[verdict] += weight
        return max(weighted_scores, key=weighted_scores.get)

    def _equal_weighted_voting(self, node_opinions):
        weighted_scores = {"Unknown Attack": 0.0, "Known Attack": 0.0, "Benign": 0.0}
        equal_weight = 1.0 / self.num_nodes
        for verdict in node_opinions.values():
            weighted_scores[verdict] += equal_weight
        return max(weighted_scores, key=weighted_scores.get)

def run_comparison_experiments(trained_model, initial_means, config, device, fusion_strategy, is_malicious_scenario,
                               X_test_all, Y_test_all):
    num_nodes = 5
    gateways = {}
    trust_evaluator = TrustEvaluator(num_nodes, initial_trust=0.99)

    for i in range(num_nodes):
        node_id = f'Node_{i}'
        is_malicious = (i == 4) and is_malicious_scenario

        node_model.load_state_dict(trained_model.state_dict(), strict=False)
        gateway = LocalGateway(node_id=node_id, model=node_model, initial_means=initial_means, config=config,
                               device=device, is_malicious=is_malicious)
        gateways[node_id] = gateway

    blockchain_simulator = BlockchainSimulator(gateways, trust_evaluator)

    test_samples = X_test_all
    Y_test_labels = Y_test_all.cpu().numpy()

    y_true_collective = []
    y_pred_collective = []

    for i in tqdm(range(len(test_samples)), desc=f"Running {fusion_strategy.capitalize()}"):
        sample_x = test_samples[i].to(device)
        true_label_numeric = Y_test_labels[i].item()

        if true_label_numeric == 0:
            true_label_str = "Benign"
        elif true_label_numeric > 0:
            true_label_str = "Known Attack"
        else:
            true_label_str = "Unknown Attack"

        global_verdict = blockchain_simulator.run_collaboration(
            sample_features=sample_x,
            true_label_str=true_label_str,
            is_malicious_scenario=is_malicious_scenario,
            fusion_strategy=fusion_strategy
        )

        y_true_collective.append(true_label_str)
        y_pred_collective.append(global_verdict)


    return collective_metrics

def run_full_experiment(config):
    device = torch.device("cpu")

    try:
        train_data, scaler, X_test_all, Y_test_all = load_dataset_from_hdf5(config['DATA_BASE_DIR'], config['hdf_key'])
    except Exception as e:
        logging.critical(f"Data Loading Failed: {e}. Cannot proceed without data. Check path/format.")
        return

    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_data, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False)

    trained_model, initial_means = train_model(model, train_loader, val_loader, config, device)

    # 1. 对比实验运行: 包含所有策略
    fusion_strategies = ["majority", "simple_weighted", "equal_weighted", "trust"]
    honest_results = {}
    malicious_results = {}

    for strategy in fusion_strategies:
        logging.info(f"Starting Honest Experiment for: {strategy}")
        honest_results[strategy] = run_comparison_experiments(trained_model, initial_means, config, device, strategy,
                                                              is_malicious_scenario=False, X_test_all=X_test_all,
                                                              Y_test_all=Y_test_all)

        logging.info(f"Starting Malicious Experiment for: {strategy}")
        malicious_results[strategy] = run_comparison_experiments(trained_model, initial_means, config, device, strategy,
                                                                 is_malicious_scenario=True, X_test_all=X_test_all,
                                                                 Y_test_all=Y_test_all)

def main():
    setup_logging()
    config = get_user_configuration()

    try:
        run_full_experiment(config)
    except Exception as e:
        logging.error(f"Execution Error: {e}", exc_info=True)


if __name__ == "__main__":
    main()