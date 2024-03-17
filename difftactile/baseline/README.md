# Baseline of DiffTactile

This folder includes files to run baseline of DiffTactile. including cma based on cma and RL based on stable-baselines3. Moreover, it includes code which used to train rnn model and evaluate model as depicted in paper.

## File Structure
* cma_baseline.py
    * Used to run CMA-ES baseline
* register_rasks.py
    * Used to register all four implemented tasks in DiffTactile, including all their parameters to run baseline.
* rnn_deploy.py
    * Deploy the trained model of RNN and predict.
* rnn_train.py
    * Used to train RNN model.
* tactile_env.py
    * This file defines environments used to run training based on stable-baselines3
* tactile_eval_env.py
    * This file inherents class defined by tactile_env.py and defines env used to eval in stable-baselines3.
* training_rl.py
    * Includes code to run SAC and PPO baseline for tasks.

## Running Method
* CMA-ES baseline
    * You can run with following command line, args parameters can be changed at will: python cma_baseline.py --use_state --use_tactile --render --task surface_follow
* RL baseline
    * You can run with following command line, args parameters can be changed at will: python training_rl.py --num_train_envs 2 --num_eval_envs 1 --max_iteration 2 --algorithm PPO --obs tactile --loss tactile --task surface_follow