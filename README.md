# AlphaZero Connect4
# From-scratch implementation of AlphaZero for Connect4

This repo demonstrates an implementation of AlphaZero framework for Connect4, using python and PyTorch.

For more implementation details, please see my published article: https://towardsdatascience.com/from-scratch-implementation-of-alphazero-for-connect4-f73d4554002a

# Contents
In this repository, you will find the following core scripts:

1) MCTS_c4.py - implements the Monte-Carlo Tree Search (MCTS) algorithm based on Polynomial Upper Confidence Trees (PUCT) method for leaf transversal. This generates datasets (state, policy, value) for neural network training

2) alpha_net_c4.py - PyTorch implementation of the AlphaZero neural network architecture, with slightly reduced number of residual blocks (19) and convolution channels (128) for faster computation. The network consists of, in order:
- A convolution block with batch normalization
- 19 residual blocks with each block consisting of two convolutional layers with batch normalization
- An output block with two heads: a policy output head that consists of convolutional layer with batch normalization followed by logsoftmax, and a value head that consists of a convolutional layer with relu and tanh activation.

3) connect_board.py – Implementation of a Connect4 board python class with all game rules and possible moves

4) encoder_decoder_c4.py – list of functions to encode/decode Connect4 board class for input/interpretation into neural network

5) evaluator_c4.py – arena class to pit current neural net against the neural net from previous iteration, and keeps the neural net that wins the most games

6) train_c4.py – function to start the neural network training process

7) visualize_board_c4.py – miscellaneous function to visualize the board in a more attractive way

8) play_against_c4.py - run it to play a Connect4 game against AlphaZero! (change "best_net" to the alpha net you've trained)

# Iteration pipeline

A full iteration pipeline consists of:
1) Self-play using MCTS (MCTS_c4.py) to generate game datasets (game state, policy, value), with the neural net guiding the search by providing the prior probabilities in the PUCT algorithm

2) Train the neural network (train_c4.py) using the (game state, policy, value) datasets generated from MCTS self-play

3) Evaluate (evaluator_c4.py) the trained neural net (at predefined checkpoints) by pitting it against the neural net from the previous iteration, again using MCTS guided by the respective neural nets, and keep only the neural net that performs better.

4) Rinse and repeat. Note that in the paper, all these processes are running simultaneously in parallel, subject to available computing resources one has.

# How to run

1) Clone the repo, then run main_pipeline.py with appropriate arguments to start training your model.
```bash
main_pipeline.py [-h] 
		[--iteration ITERATION]  
		[--total_iterations TOTAL_ITERATIONS]  
		[--MCTS_num_processes MCTS_NUM_PROCESSES]
		[--num_games_per_MCTS_process NUM_GAMES_PER_MCTS_PROCESS]  
		[--temperature_MCTS TEMPERATURE_MCTS]  
		[--num_evaluator_games NUM_EVALUATOR_GAMES]  
		[--neural_net_name NEURAL_NET_NAME]  
		[--batch_size BATCH_SIZE]  
		[--num_epochs NUM_EPOCHS]  
		[--lr LR]  
		[--gradient_acc_steps GRADIENT_ACC_STEPS]  
		[--max_norm MAX_NORM]  
```

# Results

Iteration 0:
alpha_net_0 (Initialized with random weights)
151 games of MCTS self-play generated

Iteration 1:
alpha_net_1 (trained from iteration 0)
148 games of MCTS self-play generated

Iteration 2:
alpha_net_2 (trained from iteration 1)
310 games of MCTS self-play generated

Evaluation 1:
After Iteration 2, alpha_net_2 is pitted against alpha_net_0 to check if the neural net is improving in terms of policy and value estimate. Indeed, out of 100 games played, alpha_net_2 won 83. 

Iteration 3:
alpha_net_3 (trained from iteration 2)
584 games of MCTS self-play generated

Iteration 4:
alpha_net_4 (trained from iteration 3)
753 games of MCTS self-play generated

Iteration 5:
alpha_net_5 (trained from iteration 4)
1286 games of MCTS self-play generated

Iteration 6:
alpha_net_6 (trained from iteration 5)
1670 games of MCTS self-play generated



![alt text](https://github.com/plkmo/AlphaZero_Connect4/blob/master/Loss_vs_Epoch0_iter0_2019-03-12.png) Typical Loss vs Epoch when training neural net (alpha_net_0)
