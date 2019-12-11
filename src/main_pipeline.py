# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 11:01:30 2019

@author: WT
"""
from MCTS_c4 import run_MCTS
from train_c4 import train_chessnet
from evaluator_c4 import evaluate_nets
from argparse import ArgumentParser
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--iteration", type=int, default=0, help="Current iteration number")
    parser.add_argument("--MCTS_num_processes", type=int, default=1, help="Number of processes to run MCTS self-plays")
    parser.add_argument("--num_games_per_MCTS_process", type=int, default=100, help="Number of games to simulate per MCTS self-play process")
    parser.add_argument("--temperature_MCTS", type=float, default=1.1, help="Temperature for first 10 games of MCTS self-play")
    parser.add_argument("--num_evaluator_games", type=int, default=100, help="No of games to play to evaluate neural nets")
    parser.add_argument("--neural_net_name", type=str, default="cc4_current_net_", help="Name of neural net")
    parser.add_argument("--batch_size", type=int, default=10, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=200, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--gradient_acc_steps", type=int, default=1, help="Number of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    args = parser.parse_args()
    
    logger.info("Starting iteration pipeline...")
    for i in range(args.iteration, 1000):
        run_MCTS(args, start_idx=0, iteration=i)
        train_chessnet(args, iteration=i, new_optim_state=True)
        if i >= 1:
            winner = evaluate_nets(args, i, i + 1)
            counts = 0
            while (winner != (i + 1)):
                logger.info("Trained net didn't perform better, generating more MCTS games for retraining...")
                run_MCTS(args, start_idx=(counts + 1)*args.num_games_per_MCTS_process, iteration=i)
                counts += 1
                train_chessnet(args, iteration=i, new_optim_state=True)
                winner = evaluate_nets(args, i, i + 1)