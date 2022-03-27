import argparse
from datetime import datetime

from pathlib import Path


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = value

class Parser(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def parse(self) -> argparse.ArgumentParser:
        parser_ = argparse.ArgumentParser()
        #Flow Control
        parser_.add_argument('-bl', '--baseline', default=0 ,type=int, help='enable benchmark baseline if 1 - greedy"')
        parser_.add_argument('-interfere', '--interference', default=0, type=int, help='enable interference model')
        parser_.add_argument('-eeval', '--episode-eval', default=30, type=int, help='Number of episodes to evaluate vs greedy')
        parser_.add_argument('-ecompare', '--episode-compare', default=200, type=int, help='Number of episodes to train')
        parser_.add_argument('-nodes', '--nodes-number', default=3, type=int, help='Number of nodes for single agent construction topology')
        parser_.add_argument('-evalprint', '--eval-print', default=0, type=int, help='if to print evaluation stats or not')
        parser_.add_argument('-modelname', '--model-name', default="noname", type=str, help='name of the model to load')
        parser_.add_argument('-scene', '--run-scene', default="", type=str, help='scene to run, training scene')
        parser_.add_argument('-fng', '--greedy-file-name', default="", type=str, help='Greedy comparison file that DRL should be compared to.')
        parser_.add_argument('-ooo', '--outputname', default="", type=str, help='output name for file outputted by the algorithms')
        # RL Paramaters
        parser_.add_argument('-ttt', '--train-single', default=0, type=int, help='train single agent')
        parser_.add_argument('-eps', '--epsilon', default=1, type=float, help='epsilon max"')
        parser_.add_argument('-l', '--learning-rate', default=0.00003, type=float, help='Learning rate')
        parser_.add_argument('-b', '--batch-size', default=32, type=int, help='Batch size')
        parser_.add_argument('-nn', '--nn-layer', default=64, type=int, help='Number of layers for the first layer of NN')
        parser_.add_argument('-ed', '--eps-decay', default=0.001, type=float, help='epsilon decay')
        parser_.add_argument('-bss', '--times-step', default=100000, type=int, help='Number of steps to train single agent stable baseline')
        parser_.add_argument('-lca', '--load-center-agent', default=0, type=int, help='Load a saved single stable baseline agent.')

        # ENV Parameters
        parser_.add_argument('-eee', '--eval-numbers', nargs=19, type=float, help='threshhold for interference check')
        parser_.add_argument('-fa', '--flows', default=2, type=int, help='flows generator')
        parser_.add_argument('-pa', '--packets-amount', default=10, type=int, help='packets generator')
        parser_.add_argument('-li', '--low-interference', default=0.01, type=float, help='low interference noise')
        parser_.add_argument('-hi', '--high-interference', default=0.02, type=float, help='high interference noise')
        parser_.add_argument('-ch', '--capacity-per-link-high', default=5, type=int, help='high link capacity for the system')
        parser_.add_argument('-cl', '--capacity-per-link-low', default=15, type=int, help='low link capacity for the system')
        parser_.add_argument('-lcf', '--low-count-flows', default=1, type=int, help='low count for flows from source to dest')
        parser_.add_argument('-hcf', '--high-count-flows', default=5, type=int,help='high count for flows from source to dest')
        parser_.add_argument('-lcp', '--low-count-packets', default=5, type=int,help='low count for packets in a flow ')
        parser_.add_argument('-hcp', '--high-count-packets', default=25, type=int, help='high count for packets in a flow ')
        parser_.add_argument('-trans', '--transceiver-count', default=1, type=int, help='transceiver per router ')
        parser_.add_argument('-sce', '--max-step-per-episode', default=5000, type=int, help='maximum steps per episode ')
        parser_.add_argument('--gather_stats', dest='gather_stats', default=1,action='store_true', help="Compute Average reward per episode (slower)")
        parser_.add_argument('-rc', '--rings-count', default=2, type=int, help='type of topology topology ')
        parser_.add_argument('-load', '--load-weights', default=0, type=int, help='1 to load saved models for multi agents ')
        parser_.add_argument('-qb', '--queue-size', default=650, type=int, help='queues buffer')

        args = parser_.parse_args()

        return args

