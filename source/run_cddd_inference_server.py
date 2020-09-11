import os
import logging
import time
from cddd.inference import InferenceServer
import argparse

logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(num_servers: int):
    """
    runs the inference server in an endless loop. This script should be executed in a several linux console, e.g tmux session
    :return:
    """

    import os
    source_path = os.path.abspath(__file__)
    base_path = os.path.dirname(os.path.dirname(source_path))
    model_dir = os.path.join(base_path, 'cddd/default_model')
    inference_server = InferenceServer(num_servers=num_servers, maximum_iterations=150,
                                       port_frontend=5527, port_backend=5528,
                                       model_dir=model_dir)
    while True:
        time.sleep(10)

    return None

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script to spin-up CDDD inference server and allocate processes to selected GPUs")
    parser.add_argument('--devices', action='store', dest='devices',
                        default="6,7",
                        help='Which GPU device to use. If more GPUs should be used, simply add by appending.\
                         E.G --devices 1,2 for using GPU device 1 and 2. Default: 6,7', type=str)
    parser.add_argument('--nservers', action='store', dest='nservers',
                        help='Number of CDDD servers to spin-up. Each GPU can spin up 2 servers. Default: 4 servers', type=int,
                        default=4)

    args = parser.parse_args()

    ngpus = len(args.devices.split(","))
    num_servers_should = 2*ngpus
    if args.nservers != num_servers_should:
        print("Selected number of servers is: {}. You should consider using 2*{}={} servers!".format(
            args.nservers, ngpus, num_servers_should
        ))

    print("Using GPU devices: {}".format(args.devices))
    print("Total number of servers to spin up: {}".format(args.nservers))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.devices)
    num_servers = args.nservers

    main(num_servers)
