import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1.0, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=50, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.0005, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=3, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='5',help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=64, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--hyper', type=float, default=0.3, help='hypermeter alpha')
    parser.add_argument('--dataset', type=str, default='TON_IoT', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=2, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=44, help="number of channels of TON_IoT")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log_level', type=str, default='DEBUG', help='level of logs: DEBUG, INFO, WARNING, ERROR, '
                                                                       'or CRITICAL')
    parser.add_argument('--fade', type=float, default=-1, help="static fade coefficient, -1 means dynamic")
    parser.add_argument('--dataset_train_size', type=int, default=46484, help="total dataset training size")
    parser.add_argument('--test_ip_addr', type=str, default="10.150.187.13", help="ip address used to test local IP")
    parser.add_argument('--start_sleep', type=int, default=60, help="sleep for seconds before start train")
    parser.add_argument('--exit_sleep', type=int, default=60, help="sleep for seconds before exit python")
    parser.add_argument("--poisoning_attackers", nargs="+", default=[])
    parser.add_argument("--poisoning_detect_threshold", type=float, default=0.8)
    parser.add_argument("--ddos_duration", type=int, default=-1)
    parser.add_argument("--ddos_no_response_percent", type=float, default=0.9)

    args = parser.parse_args()
    return args
