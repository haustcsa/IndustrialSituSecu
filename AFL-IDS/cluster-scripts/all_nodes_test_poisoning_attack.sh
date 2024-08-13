#!/bin/bash

# set -x

source ./test.config
source ../fabric-network/network.config

function killOldProcesses() {
    # kill all old processes
    ./stop_fed_async.sh
    ./stop_fed_sync.sh
    ./stop_fed_avg.sh
    ./stop_fed_localA.sh
    ./stop_local_train.sh
    ./stop_fed_bdfl.sh
    ./stop_fed_asofed.sh
}

function cleanOutput() {
    # clean all old outputs
    ./clean-output.sh
}

function clean() {
    killOldProcesses
    cleanOutput
}

function arrangeOutput(){
    model=$1
    dataset=$2
    expname=$3
    ./gather-output.sh
    mkdir -p "${model}-${dataset}"
    mv output/ "${model}-${dataset}/${expname}"
}


function testFinish() {
    fileName=$1
    while : ; do
        count=$(ps -ef|grep ${fileName}|wc -l)
        if [[ $count -eq 0 ]]; then
            break
        fi
        echo "[`date`] Process still active, sleep 60 seconds"
        sleep 60
    done
}


function main() {
    for i in "${!TestSchema[@]}"; do
        schema=(${TestSchema[i]//-/ })
        model=${schema[0]}
        dataset=${schema[1]}
        is_iid=${IS_IID}
        attackers=${POISONING_ATTACKER}
        echo "[`date`] ALL_NODE_TEST UNDER: ${model} - ${dataset}"

        if [[ ! -d "${model}-${dataset}/fed_async_defense" ]]; then
            echo "[`date`] ## fed_async_defense start ##"
            # clean
            clean
            # run test
            for i in {0..4}; do
              num=$((i + 1))
              addrIN="127.0.0.1"
              dataset_train_size=${TrainDataSize[i]}
              ./restart_core.sh ${HostUser} ${addrIN[0]} "fed_async_${num}" "$model" "$dataset" "$is_iid" "$dataset_train_size" "-1" "$attackers" "0.8"
            done

            sleep 60
            curl -i -X POST -H "Content-Type: application/json" -d '{"message":"prepare","uuid":"1","epochs":30}' http://localhost:8888/messages
            curl -i -X POST -H "Content-Type: application/json" -d '{"message":"prepare","uuid":"2","epochs":30}' http://localhost:8889/messages
            curl -i -X POST -H "Content-Type: application/json" -d '{"message":"prepare","uuid":"3","epochs":30}' http://localhost:8890/messages
            curl -i -X POST -H "Content-Type: application/json" -d '{"message":"prepare","uuid":"4","epochs":30}' http://localhost:8891/messages
            curl -i -X POST -H "Content-Type: application/json" -d '{"message":"prepare","uuid":"5","epochs":30}' http://localhost:8892/messages

            # detect test finish or not
            testFinish "[f]ed_async_1.py"
            testFinish "[f]ed_async_2.py"
            testFinish "[f]ed_async_3.py"
            testFinish "[f]ed_async_4.py"
            testFinish "[f]ed_async_5.py"

            # gather output, move to the right directory
            arrangeOutput ${model} ${dataset} "fed_async_defense"
            echo "[`date`] ## fed_async_defense done ##"
        fi


    done
}

main > full_test.log 2>&1 &
