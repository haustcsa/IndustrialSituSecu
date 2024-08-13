#!/bin/bash

source ../fabric-network/network.config

for i in {0..19}; do
  num=$((i + 1))
  addrIN="127.0.0.1"
  ./stop_core.sh ${HostUser} ${addrIN[0]}  "fed_async_${num}"
done


