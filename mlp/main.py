import argparse
import logging
import os

from tqdm import tqdm

import torch
import torch.nn.functional as F

import wandb

from mlp import MLP
from generate_data import generate_batch_items, generate_batch_trials

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", type=int, required=True)
    parser.add_argument("--num_episodes", type=int, required=True)
    parser.add_argument("--num_train_trials", type=int, required=True)
    parser.add_argument("--num_test_trials", type=int, required=True)
    parser.add_argument("--num_items", type=int, required=True)
    parser.add_argument("--item_size", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--grad_clip", type=float, required=True)
    parser.add_argument("--num_episodes_per_reset", type=int, required=True)
    return parser.parse_args()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"args: {args}")
    num_episodes = args.num_episodes
    num_episodes_per_reset = args.num_episodes_per_reset
    num_train_trials = args.num_train_trials
    num_test_trials = args.num_test_trials

    num_items = args.num_items
    item_size = args.item_size
    batch_size = args.batch_size

    input_size = (2 * item_size) + 2  # 2 additional neurons for the previous reward and choice

    wandb.init(project="3factor", name=f"mlp_{args.hidden_size}_{args.learning_rate}")

    model = MLP(input_size, args.hidden_size, args.batch_size).to(device)
    wandb.watch(model, log="all", log_freq=100)
    logger.info(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-6)

    for episode in range(num_episodes):
        if episode % num_episodes_per_reset == 0:
            plastic_weights = torch.zeros(batch_size, args.hidden_size, args.hidden_size, dtype=torch.float32, requires_grad=False).to(device)
        else:
            plastic_weights = plastic_weights.detach()

        batch_items = generate_batch_items(num_items, item_size, batch_size)

        trials, correct_choices = generate_batch_trials(batch_items, num_train_trials, num_test_trials)

        trials = torch.tensor(trials, dtype=torch.float32).to(device)
        correct_choices = torch.tensor(correct_choices, dtype=torch.float32).to(device)
        optimizer.zero_grad()
        episode_loss = torch.tensor(0.0, dtype=torch.float32).to(device)
        correct_train_choices = 0
        correct_test_choices = 0
        prev_choice_made = None
        prev_correct = None

        for trial in range(num_train_trials + num_test_trials):
            batch_trial = trials[:, trial, :]
            batch_correct_choice = correct_choices[:, trial]
            if trial == 0:
                prev_reward = torch.zeros(batch_size, 1, device=device)
                prev_choice = torch.zeros(batch_size, 1, device=device)
            else:
                prev_reward = (prev_choice_made == prev_correct).float().unsqueeze(-1) * 2 - 1  # +1 or -1
                prev_choice = prev_choice_made.unsqueeze(-1)
            trial_input = torch.cat([batch_trial, prev_reward, prev_choice], dim=-1)

            choice, neuromodulator, value, plastic_weights, hidden = model(trial_input, plastic_weights)

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
        if episode > 100:  # Burn-in period
            optimizer.step()

        wandb.log({
            "episode_loss": episode_loss,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
        })


if __name__ == "__main__":
    args = parse_args()
    main(args)
