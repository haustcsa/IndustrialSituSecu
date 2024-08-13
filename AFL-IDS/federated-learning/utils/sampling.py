#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np

# 设置随机种子
seed = 42
np.random.seed(seed)

def get_indices(labels, user_labels, n_samples):
    indices = []
    for selected_label in user_labels:
        label_samples = np.where(labels[1, :] == selected_label)
        label_indices = labels[0, label_samples]
        selected_indices = list(np.random.choice(label_indices[0], n_samples, replace=True))
        indices += selected_indices
    return indices

def get_samples(indices, n_samples):
    selected_indices = list(np.random.choice(indices, n_samples, replace=False))
    return selected_indices

def iid_onepass(dataset_train, dataset_train_size, dataset_test, dataset_test_size, num_users, dataset_name='TON_IoT'):
    train_users = {}
    test_users = {}

    train_idxs = dataset_train.idxs

    train_labels = dataset_train.targets
    train_labels = np.vstack((train_idxs, train_labels))
    test_idxs = dataset_test.idxs
    test_labels = dataset_test.targets
    test_labels = np.vstack((test_idxs, test_labels))

    if dataset_name == 'TON_IoT':
        data_classes = 2
        # data_classes = 9
    elif dataset_name == 'Gas':
        data_classes = 2
        # data_classes = 7
    else:
        data_classes = 0

    labels = list(range(data_classes))
    train_samples = int(dataset_train_size / data_classes)
    test_indices = test_idxs

    for i in range(num_users):
        train_indices = get_indices(train_labels, labels, n_samples=train_samples)
        train_users[i] = train_indices
        test_users[i] = test_indices
    return train_users, test_users
