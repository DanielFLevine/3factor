import argparse
import logging
import os

from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

import wandb

from mlp import MLP
from generate_data import generate_batch_items, generate_batch_trials_ti, generate_batch_trials_ll
from plots import plot_pca_inputs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", type=int, default=200, required=False, help="Size of hidden dimension")
    parser.add_argument("--num_episodes", type=int, default=30000, required=False, help="Number of episodes to train for")
    parser.add_argument("--num_train_trials", type=int, default=64, required=False, help="Number of training trials per episode for transitive inference task")
    parser.add_argument("--num_test_trials", type=int, default=32, required=False, help="Number of test trials per episode for transitive inference task")
    parser.add_argument("--num_items", type=int, default=7, required=False, help="Number of items in transitive inference task")
    parser.add_argument("--item_size", type=int, default=32, required=False, help="Dimensionality of each item")
    parser.add_argument("--batch_size", type=int, default=32, required=False, help="Batch size or number of synchronous agents in each episode. Taken from A2C algorithm even though we don't use the policy loss")
    parser.add_argument("--learning_rate", type=float, default=0.0001, required=False, help="Learning rate for the optimizer/outer loop training")
    parser.add_argument("--grad_clip", type=float, default=2.0, required=False, help="Gradient clipping for the optimizer/outer loop training")
    parser.add_argument("--num_episodes_per_reset", type=int, default=1, required=False, help="Number of episodes per reset of plastic weights")
    parser.add_argument("--item_range", type=int, nargs='+', default=[4, 9], required=False, help="Range of number of items in each episode")
    parser.add_argument("--change_items_throughout_batch", action='store_true', required=False, help="Each agent in an episode sees a different set of item representations")
    parser.add_argument("--num_trials_list_1", type=int, required=False, help="Number of trials in list 1 for list-linking task")
    parser.add_argument("--num_trials_list_2", type=int, required=False, help="Number of trials in list 2 for list-linking task")
    parser.add_argument("--num_trials_linking_pair", type=int, required=False, help="Number of trials in linking pair for list-linking task")
    parser.add_argument("--use_ll", action='store_true', required=False, help="Perform list-linking instead of transitive inference task")
    parser.add_argument("--extra_layers", type=int, default=0, required=False, help="Number of extra hidden layers prior to final hidden layer that combines with plastic weights")
    parser.add_argument("--burn_in_period", type=int, default=100, required=False, help="Number of episodes to burn in for before training")
    return parser.parse_args()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"args: {args}")
    num_episodes = args.num_episodes
    num_episodes_per_reset = args.num_episodes_per_reset
    num_train_trials = args.num_train_trials
    num_test_trials = args.num_test_trials
    if args.use_ll:
        assert args.num_trials_list_1 is not None
        assert args.num_trials_list_2 is not None
        assert args.num_trials_linking_pair is not None
        num_train_trials = args.num_trials_list_1 + args.num_trials_list_2 + args.num_trials_linking_pair

    item_size = args.item_size
    batch_size = args.batch_size

    input_size = 2*item_size

    wandb.init(project="3factor", name=f"mlp_{args.hidden_size}_{args.learning_rate}")

    model = MLP(input_size, args.hidden_size, args.batch_size).to(device)
    wandb.watch(model, log="all", log_freq=100)
    logger.info(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-6)

    for episode in range(num_episodes):
        num_items = np.random.randint(args.item_range[0], args.item_range[1])
        if episode % num_episodes_per_reset == 0:
            plastic_weights = torch.zeros(batch_size, args.hidden_size, args.hidden_size, dtype=torch.float32, requires_grad=False).to(device)
        else:
            plastic_weights = plastic_weights.detach()

        batch_items = generate_batch_items(num_items, item_size, batch_size, change_items_throughout_batch=args.change_items_throughout_batch)

        if args.use_ll:
            trials, correct_choices = generate_batch_trials_ll(batch_items, args.num_trials_list_1, args.num_trials_list_2, args.num_trials_linking_pair, num_test_trials)
        else:
            trials, correct_choices = generate_batch_trials_ti(batch_items, num_train_trials, num_test_trials)

        trials = torch.tensor(trials, dtype=torch.float32).to(device)
        correct_choices = torch.tensor(correct_choices, dtype=torch.float32).to(device)
        optimizer.zero_grad()
        episode_loss = torch.tensor(0.0, dtype=torch.float32).to(device)
        correct_train_choices = 0
        correct_test_choices = 0

        for trial in range(num_train_trials + num_test_trials):
            batch_trial = trials[:, trial, :]
            batch_correct_choice = correct_choices[:, trial]
            
            trial_input = batch_trial

            choice, neuromodulator, value, plastic_weights, hidden = model(trial_input, plastic_weights, batch_correct_choice)

            if torch.isnan(choice).any() or (choice < 0).any() or (choice > 1).any():
                print(f"Trial {trial}: choice has invalid values - min={choice.min()}, max={choice.max()}, nan={torch.isnan(choice).sum()}")
                break

            choice_sampled = torch.bernoulli(choice).squeeze(-1)
            choice_prob = choice.squeeze(-1)
            loss = F.binary_cross_entropy(choice_prob, batch_correct_choice, reduction='sum') / ((num_train_trials + num_test_trials) * batch_size)
            episode_loss += loss

            if trial < num_train_trials:
                correct_train_choices += (choice_sampled == batch_correct_choice).sum().item()
            else:
                correct_test_choices += (choice_sampled == batch_correct_choice).sum().item()

            prev_correct = batch_correct_choice
            prev_choice_made = choice_sampled

        # l2_plastic_weights_loss = torch.mean(plastic_weights ** 2) * 1e-1
        # episode_loss += l2_plastic_weights_loss
        # alpha_reg = torch.mean(model.alpha ** 2) * 1e-1
        # episode_loss += alpha_reg

        train_accuracy = correct_train_choices / (num_train_trials * batch_size)
        test_accuracy = correct_test_choices / (num_test_trials * batch_size)

        if episode % 100 == 0:
            logger.info(f"Episode {episode}, Loss: {episode_loss}, Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
            # logger.info(f"Plastic weights loss: {l2_plastic_weights_loss}, Alpha regularization: {alpha_reg}")
            logger.info(f"Neuromodulator values: {neuromodulator}, {neuromodulator.min().item()}, {neuromodulator.max().item()}, {neuromodulator.mean().item()}")
            logger.info(f"Plastic weights: {plastic_weights.abs().mean()}, {plastic_weights.abs().min().item()}, {plastic_weights.abs().max().item()}, {plastic_weights.abs().mean().item()}")
            logger.info(f"Choice: {choice_prob}, {choice_prob.min().item()}, {choice_prob.max().item()}, {choice_prob.mean().item()}")
            logger.info(f"Contributions - plastic: {model.plastic_contribution_mag}, current: {model.current_contribution_mag}")

        episode_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if episode > args.burn_in_period:
            optimizer.step()

        wandb_log_dict = {
            "episode_loss": episode_loss,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
        }

        if episode % 100 == 0:
            fig = plot_pca_inputs(trials, model, episode)
            wandb_log_dict["pca_inputs"] = wandb.Image(fig)
        wandb.log(wandb_log_dict)


if __name__ == "__main__":
    args = parse_args()
    main(args)
