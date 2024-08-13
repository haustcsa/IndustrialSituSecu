#!/bin/bash

source ../fabric-network/network.config

rm -rf output/
mkdir -p output/

for i in "${!PeerAddress[@]}"; do
  addrIN=(${PeerAddress[i]//:/ })
  echo "copy output"
  cp $PWD/../federated-learning/result-record_*.txt output/
done

