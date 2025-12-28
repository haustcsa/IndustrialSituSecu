import os
import logging
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from datasets.CIC_IDS_2017.prepare_data import parse_configuration

def divide_dataset_into_nodes(x_train, y_train, x_test, y_test, output_dir, hdf_key, n_nodes=5):

    if not os.path.exists(output_dir):
        logging.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    for node_id in range(n_nodes):
        node_dir = os.path.join(output_dir, f"node_{node_id}")
        logging.info(f"Creating directory: {node_dir}")
        os.makedirs(node_dir, exist_ok=True)

    skf_train = StratifiedKFold(n_splits=n_nodes, shuffle=True, random_state=42)
    for node_id, (_, idx) in enumerate(skf_train.split(x_train, y_train)):
        node_x_train = x_train.iloc[idx]
        node_y_train = y_train.iloc[idx]
        node_path = os.path.join(output_dir, f"node_{node_id}")

        node_x_train.to_hdf(os.path.join(node_path, "x_train.h5"), key=hdf_key, mode="w", complevel=5)
        node_y_train.to_hdf(os.path.join(node_path, "y_train.h5"), key=hdf_key, mode="w", complevel=5)
        logging.info(f"Node {node_id} train subset saved. Samples: {len(node_x_train)}")

    skf_test = StratifiedKFold(n_splits=n_nodes, shuffle=True, random_state=24)
    for node_id, (_, idx) in enumerate(skf_test.split(x_test, y_test)):
        node_x_test = x_test.iloc[idx]
        node_y_test = y_test.iloc[idx]
        node_path = os.path.join(output_dir, f"node_{node_id}")
 
        node_x_test.to_hdf(os.path.join(node_path, "x_test.h5"), key=hdf_key, mode="w", complevel=5)
        node_y_test.to_hdf(os.path.join(node_path, "y_test.h5"), key=hdf_key, mode="w", complevel=5)
        logging.info(f"Node {node_id} test subset saved. Samples: {len(node_x_test)}")

def main():
    params = parse_configuration()

    x_train = pd.read_hdf(os.path.join(params["output_dir"], "x_train.h5"), key=params["hdf_key"])
    y_train = pd.read_hdf(os.path.join(params["output_dir"], "y_train.h5"), key=params["hdf_key"])
    x_test = pd.read_hdf(os.path.join(params["output_dir"], "x_test.h5"), key=params["hdf_key"])
    y_test = pd.read_hdf(os.path.join(params["output_dir"], "y_test.h5"), key=params["hdf_key"])

    divide_dataset_into_nodes(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        output_dir=params["output_dir"],
        hdf_key=params["hdf_key"],
        n_nodes=5
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
