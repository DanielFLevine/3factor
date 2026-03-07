import torch
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_bce_loss(
    choice_probs,
    correct_choices,
    nonadjacent_mask,
    num_train_trials,
    num_test_trials,
    batch_size,
    nonadj_loss_multiplier=1.0,
    mask_adjacent_loss=False,
    task_labels=None,
):
    """
    Compute BCE loss for the episode.

    Args:
        choice_probs: list of tensors, each of shape (batch_size,) - predicted probabilities per trial
        correct_choices: tensor of shape (batch_size, num_trials) - correct answers
        nonadjacent_mask: tensor of shape (batch_size, num_trials) - mask for non-adjacent pairs
        num_train_trials: number of training trials
        num_test_trials: number of test trials
        batch_size: batch size
        nonadj_loss_multiplier: multiplier for non-adjacent pair loss
        mask_adjacent_loss: if True, only compute loss on non-adjacent pairs (adjacent pairs contribute 0 to loss)
        task_labels: optional tensor of shape (batch_size, num_trials) - 0 for TI, 1 for AI (for interleaved mode)

    Returns:
        episode_loss: scalar tensor
        loss_dict: dictionary with loss breakdown (train, test, nonadjacent, and optionally ti_*/ai_* for interleaved)
    """
    device = choice_probs[0].device
    episode_loss = torch.tensor(0.0, device=device)
    num_trials = len(choice_probs)

    # Track losses by category (unweighted, for logging)
    train_loss_sum = torch.tensor(0.0, device=device)
    test_loss_sum = torch.tensor(0.0, device=device)
    nonadj_loss_sum = torch.tensor(0.0, device=device)
    train_count = 0
    test_count = 0
    nonadj_count = 0

    # Task-specific loss tracking for interleaved mode
    if task_labels is not None:
        ti_train_loss_sum = torch.tensor(0.0, device=device)
        ti_test_loss_sum = torch.tensor(0.0, device=device)
        ai_train_loss_sum = torch.tensor(0.0, device=device)
        ai_test_loss_sum = torch.tensor(0.0, device=device)
        ti_train_count = 0
        ti_test_count = 0
        ai_train_count = 0
        ai_test_count = 0

    for trial in range(num_trials):
        choice_prob = choice_probs[trial]
        batch_correct_choice = correct_choices[:, trial]
        batch_nonadjacent = nonadjacent_mask[:, trial].float()

        nonreduced_loss = F.binary_cross_entropy(choice_prob, batch_correct_choice, reduction='none')

        if mask_adjacent_loss:
            # Only compute loss on non-adjacent pairs (adjacent pairs masked out)
            weight = batch_nonadjacent * nonadj_loss_multiplier
        else:
            # Weight: 1.0 for adjacent trials, nonadj_loss_multiplier for non-adjacent trials
            weight = 1.0 + batch_nonadjacent * (nonadj_loss_multiplier - 1.0)

        loss = nonreduced_loss * weight
        loss = loss.sum() / ((num_train_trials + num_test_trials) * batch_size)
        episode_loss += loss

        # Track unweighted losses for logging
        if trial < num_train_trials:
            train_loss_sum += nonreduced_loss.mean()
            train_count += 1
        else:
            test_loss_sum += nonreduced_loss.mean()
            test_count += 1

        # Track nonadjacent loss (for trials where there are nonadjacent pairs)
        nonadj_mask_bool = batch_nonadjacent > 0.5
        if nonadj_mask_bool.any():
            nonadj_loss_sum += nonreduced_loss[nonadj_mask_bool].mean()
            nonadj_count += 1

        # Task-specific loss tracking for interleaved mode
        if task_labels is not None:
            trial_task_labels = task_labels[:, trial]
            ti_mask = trial_task_labels == 0
            ai_mask = trial_task_labels == 1

            if ti_mask.any():
                ti_loss = nonreduced_loss[ti_mask].mean()
                if trial < num_train_trials:
                    ti_train_loss_sum += ti_loss
                    ti_train_count += 1
                else:
                    ti_test_loss_sum += ti_loss
                    ti_test_count += 1

            if ai_mask.any():
                ai_loss = nonreduced_loss[ai_mask].mean()
                if trial < num_train_trials:
                    ai_train_loss_sum += ai_loss
                    ai_train_count += 1
                else:
                    ai_test_loss_sum += ai_loss
                    ai_test_count += 1

    # Compute average losses for logging
    loss_dict = {
        'train_loss': (train_loss_sum / train_count).item() if train_count > 0 else 0.0,
        'test_loss': (test_loss_sum / test_count).item() if test_count > 0 else 0.0,
        'nonadj_loss': (nonadj_loss_sum / nonadj_count).item() if nonadj_count > 0 else 0.0,
    }

    # Add task-specific losses for interleaved mode
    if task_labels is not None:
        loss_dict.update({
            'ti_train_loss': (ti_train_loss_sum / ti_train_count).item() if ti_train_count > 0 else 0.0,
            'ti_test_loss': (ti_test_loss_sum / ti_test_count).item() if ti_test_count > 0 else 0.0,
            'ai_train_loss': (ai_train_loss_sum / ai_train_count).item() if ai_train_count > 0 else 0.0,
            'ai_test_loss': (ai_test_loss_sum / ai_test_count).item() if ai_test_count > 0 else 0.0,
        })

    return episode_loss, loss_dict


def compute_a2c_loss(
    choice_probs,
    sampled_choices,
    values,
    correct_choices,
    nonadjacent_mask,
    num_train_trials,
    gamma=0.9,
    value_loss_coef=0.1,
    entropy_coef=0.1,
    nonadj_loss_multiplier=1.0,
    use_sos_entropy=False,
    mask_adjacent_loss=False,
):
    """
    Compute A2C loss for the episode.

    Args:
        choice_probs: list of tensors, each of shape (batch_size,) - predicted probabilities per trial
        sampled_choices: list of tensors, each of shape (batch_size,) - sampled actions per trial
        values: list of tensors, each of shape (batch_size, 1) - value predictions per trial
        correct_choices: tensor of shape (batch_size, num_trials) - correct answers
        nonadjacent_mask: tensor of shape (batch_size, num_trials) - mask for non-adjacent pairs
        num_train_trials: number of training trials
        gamma: discount factor for temporal discounting
        value_loss_coef: coefficient for value loss
        entropy_coef: coefficient for entropy bonus/penalty
        nonadj_loss_multiplier: multiplier for non-adjacent pair loss
        use_sos_entropy: if True, use sum-of-squares as entropy proxy (like Miconi);
                         if False, use actual entropy
        mask_adjacent_loss: if True, only compute loss on non-adjacent pairs (adjacent pairs contribute 0 to loss)

    Returns:
        total_loss: scalar tensor
        loss_dict: dictionary with individual loss components for logging
    """
    device = choice_probs[0].device
    num_trials = len(choice_probs)
    batch_size = choice_probs[0].shape[0]

    # Compute rewards for each trial: +1 if correct, -1 if incorrect
    rewards = []
    for trial in range(num_trials):
        batch_correct_choice = correct_choices[:, trial]
        batch_sampled = sampled_choices[trial]
        # Reward is +1 for correct, -1 for incorrect
        reward = 2.0 * (batch_sampled == batch_correct_choice).float() - 1.0
        rewards.append(reward)

    # Compute discounted returns (backward pass)
    returns = []
    R = torch.zeros(batch_size, device=device)
    for trial in reversed(range(num_trials)):
        R = rewards[trial] + gamma * R
        returns.insert(0, R.clone())

    # Compute log probabilities and entropy/sos terms
    log_probs = []
    entropy_terms = []
    for trial in range(num_trials):
        choice_prob = choice_probs[trial]
        batch_sampled = sampled_choices[trial]

        # Clamp probabilities to avoid log(0)
        choice_prob_clamped = torch.clamp(choice_prob, min=1e-7, max=1-1e-7)

        # Log probability of the sampled action
        # If sampled = 1: log_prob = log(choice_prob)
        # If sampled = 0: log_prob = log(1 - choice_prob)
        log_prob = batch_sampled * torch.log(choice_prob_clamped) + (1 - batch_sampled) * torch.log(1 - choice_prob_clamped)
        log_probs.append(log_prob)

        if use_sos_entropy:
            # Sum-of-squares as entropy proxy (like Miconi)
            # This penalizes confident predictions (pushes toward 0.5)
            # Miconi: loss += bent * y.pow(2).sum() / BS
            # For Bernoulli with prob p, we use p^2 + (1-p)^2
            sos = choice_prob ** 2 + (1 - choice_prob) ** 2
            entropy_terms.append(sos)
        else:
            # Actual entropy of Bernoulli distribution: -p*log(p) - (1-p)*log(1-p)
            entropy = -(choice_prob_clamped * torch.log(choice_prob_clamped) + (1 - choice_prob_clamped) * torch.log(1 - choice_prob_clamped))
            entropy_terms.append(entropy)

    # Compute advantages and losses
    policy_loss = torch.tensor(0.0, device=device)
    value_loss = torch.tensor(0.0, device=device)
    entropy_term_total = torch.tensor(0.0, device=device)

    for trial in range(num_trials):
        batch_nonadjacent = nonadjacent_mask[:, trial].float()
        value = values[trial].squeeze(-1)
        advantage = returns[trial] - value.detach()

        if mask_adjacent_loss:
            # Only compute loss on non-adjacent pairs (adjacent pairs masked out)
            weight = batch_nonadjacent * nonadj_loss_multiplier
        else:
            # Weight: 1.0 for adjacent trials, nonadj_loss_multiplier for non-adjacent trials
            weight = 1.0 + batch_nonadjacent * (nonadj_loss_multiplier - 1.0)

        # Policy loss: -log_prob * advantage
        trial_policy_loss = -(log_probs[trial] * advantage * weight).mean()
        policy_loss += trial_policy_loss

        # Value loss: MSE between predicted value and actual return
        trial_value_loss = ((value - returns[trial].detach()) ** 2 * weight).mean()
        value_loss += trial_value_loss

        # Entropy term
        trial_entropy_term = (entropy_terms[trial] * weight).mean()
        entropy_term_total += trial_entropy_term

    # Normalize by number of trials
    policy_loss = policy_loss / num_trials
    value_loss = value_loss / num_trials
    entropy_term_total = entropy_term_total / num_trials

    # Total loss
    # For SoS entropy: we ADD it (penalizes confident predictions)
    # For actual entropy: we SUBTRACT it (rewards high entropy / exploration)
    if use_sos_entropy:
        total_loss = policy_loss + value_loss_coef * value_loss + entropy_coef * entropy_term_total
    else:
        total_loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy_term_total

    loss_dict = {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy_term': entropy_term_total.item(),
    }

    return total_loss, loss_dict
