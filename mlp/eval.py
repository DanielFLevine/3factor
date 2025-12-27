import numpy as np
import torch

from matplotlib import pyplot as plt

from generate_data import generate_batch_items, generate_batch_trials_ti

def update_symbolic_distance_bookkeeping(symbolic_distance_bookkeeping, pair_indices, choice_sampled, correct_choices, episode):
    batch_size = pair_indices.shape[0]
    num_trials = pair_indices.shape[1]

    for batch_index in range(batch_size):
        # Build list of trial dicts for this episode
        episode_trials = []
        for trial_num in range(num_trials):
            episode_trials.append({
                "item_1": int(pair_indices[batch_index][trial_num][0]),
                "item_2": int(pair_indices[batch_index][trial_num][1]),
                "model_output": int(choice_sampled[batch_index][trial_num]),
                "correct_choice": int(correct_choices[batch_index][trial_num]),
            })
        # Append this episode's trials to the batch's bookkeeping
        symbolic_distance_bookkeeping[batch_index].append(episode_trials)

    return symbolic_distance_bookkeeping

def more_items_generalization_test(args, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    additional_items = args.additional_items
    max_training_items = args.item_range[-1] - 1
    extended_num_train_trials = args.num_train_trials
    extended_num_test_trials = args.num_test_trials
    batch_size = args.batch_size
    item_size = args.item_size

    length_generalization_logging_dict = {}

    plot_x_values = []
    plot_train_accuracies = []
    plot_test_accuracies = []

    for i in range(1, additional_items + 1):

        plastic_weights = torch.zeros(batch_size, args.hidden_size, args.hidden_size, dtype=torch.float32, requires_grad=False).to(device)

        num_items = max_training_items + i
        plot_x_values.append(num_items)
        extended_num_train_trials = extended_num_train_trials + 2
        extended_num_test_trials = extended_num_test_trials + 2
        batch_items = generate_batch_items(num_items, item_size, batch_size, change_items_throughout_batch=args.change_items_throughout_batch)

        trials, correct_choices, pair_indices = generate_batch_trials_ti(batch_items, extended_num_train_trials, extended_num_test_trials, arbitrary=args.arbitrary)

        trials = torch.tensor(trials, dtype=torch.float32).to(device)
        correct_choices = torch.tensor(correct_choices, dtype=torch.float32).to(device)
        correct_train_choices = 0
        correct_test_choices = 0
        all_choices_sampled = []

        for trial in range(extended_num_train_trials + extended_num_test_trials):
            batch_trial = trials[:, trial, :]
            batch_correct_choice = correct_choices[:, trial]
            
            trial_input = batch_trial

            with torch.inference_mode():
                choice, neuromodulator, value, plastic_weights, hidden = model(trial_input, plastic_weights, batch_correct_choice)

            if torch.isnan(choice).any() or (choice < 0).any() or (choice > 1).any():
                print(f"Trial {trial}: choice has invalid values - min={choice.min()}, max={choice.max()}, nan={torch.isnan(choice).sum()}")
                break

            choice_sampled = torch.bernoulli(choice).squeeze(-1)
            all_choices_sampled.append(choice_sampled)

            if trial < extended_num_train_trials:
                correct_train_choices += (choice_sampled == batch_correct_choice).sum().item()
            else:
                correct_test_choices += (choice_sampled == batch_correct_choice).sum().item()
        
        # Stack choices: shape (num_trials, batch_size) -> transpose to (batch_size, num_trials)
        all_choices_sampled = torch.stack(all_choices_sampled, dim=0).T.detach().cpu().numpy()

        train_accuracy = correct_train_choices / (extended_num_train_trials * batch_size)
        test_accuracy = correct_test_choices / (extended_num_test_trials * batch_size)
        plot_train_accuracies.append(train_accuracy)
        plot_test_accuracies.append(test_accuracy)

        length_generalization_logging_dict.update({
            f"length_generalize/{num_items}_item_generalize_train_accuracy": train_accuracy,
            f"length_generalize/{num_items}_item_generalize_test_accuracy": test_accuracy,
        })

    plt.figure(dpi=300)
    plt.plot(plot_x_values, plot_train_accuracies, label='Train', color='blue')
    plt.plot(plot_x_values, plot_test_accuracies, label='Test', color='green')
    plt.xlabel('Number of Items')
    plt.ylabel('Generalization Accuracy')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    fig = plt.gcf()
    plt.close()
    model.train()
    
    return length_generalization_logging_dict, fig

def mass_presentation_test(args, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    num_items = args.item_range[-1] - 1
    num_train_trials = args.num_train_trials
    num_test_trials = args.num_test_trials
    batch_size = args.batch_size
    item_size = args.item_size

    plot_train_accuracies = []
    plot_test_accuracies = []

    mass_presentation_logging_dict = {}
    mass_presentation_counts = [10*(i+1) for i in range(10)]

    for mass_presentation_count in mass_presentation_counts:

        plastic_weights = torch.zeros(batch_size, args.hidden_size, args.hidden_size, dtype=torch.float32, requires_grad=False).to(device)

        batch_items = generate_batch_items(num_items, item_size, batch_size, change_items_throughout_batch=args.change_items_throughout_batch)

        trials, correct_choices, pair_indices = generate_batch_trials_ti(batch_items, num_train_trials, num_test_trials, arbitrary=args.arbitrary, mass_presentation=mass_presentation_count)

        trials = torch.tensor(trials, dtype=torch.float32).to(device)
        correct_choices = torch.tensor(correct_choices, dtype=torch.float32).to(device)
        correct_train_choices = 0
        correct_test_choices = 0
        all_choices_sampled = []

        total_trial_count = num_train_trials + mass_presentation_count + num_test_trials

        for trial in range(total_trial_count):
            batch_trial = trials[:, trial, :]
            batch_correct_choice = correct_choices[:, trial]
            
            trial_input = batch_trial

            with torch.inference_mode():
                choice, neuromodulator, value, plastic_weights, hidden = model(trial_input, plastic_weights, batch_correct_choice)

            if torch.isnan(choice).any() or (choice < 0).any() or (choice > 1).any():
                print(f"Trial {trial}: choice has invalid values - min={choice.min()}, max={choice.max()}, nan={torch.isnan(choice).sum()}")
                break

            choice_sampled = torch.bernoulli(choice).squeeze(-1)
            all_choices_sampled.append(choice_sampled)

            if trial < num_train_trials+mass_presentation_count:
                correct_train_choices += (choice_sampled == batch_correct_choice).sum().item()
            else:
                correct_test_choices += (choice_sampled == batch_correct_choice).sum().item()
        
        # Stack choices: shape (num_trials, batch_size) -> transpose to (batch_size, num_trials)
        all_choices_sampled = torch.stack(all_choices_sampled, dim=0).T.detach().cpu().numpy()

        train_accuracy = correct_train_choices / ((num_train_trials + mass_presentation_count) * batch_size)
        test_accuracy = correct_test_choices / (num_test_trials * batch_size)
        plot_train_accuracies.append(train_accuracy)
        plot_test_accuracies.append(test_accuracy)

        mass_presentation_logging_dict.update({
            f"mass_presentation/{mass_presentation_count}_mass_presentation_train_accuracy": train_accuracy,
            f"mass_presentation/{mass_presentation_count}_mass_presentation_test_accuracy": test_accuracy,
        })

    plt.figure(dpi=300)
    plt.plot(mass_presentation_counts, plot_train_accuracies, label='Train (with mass_presentation)', color='blue')
    plt.plot(mass_presentation_counts, plot_test_accuracies, label='Test', color='green')
    plt.xlabel('Mass Presentation Count')
    plt.ylabel('Accuracy')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    fig = plt.gcf()
    plt.close()
    model.train()
    
    return mass_presentation_logging_dict, fig

def new_items_old_items_test(args, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    num_items = args.item_range[-1] - 1
    num_train_trials = args.num_train_trials
    num_test_trials = args.num_test_trials
    batch_size = args.batch_size
    item_size = args.item_size

    plastic_weights = torch.zeros(batch_size, args.hidden_size, args.hidden_size, dtype=torch.float32, requires_grad=False).to(device)

    batch_items_old = generate_batch_items(num_items, item_size, batch_size, change_items_throughout_batch=args.change_items_throughout_batch)
    batch_items_new = generate_batch_items(num_items, item_size, batch_size, change_items_throughout_batch=args.change_items_throughout_batch)

    trials_old, correct_choices_old, pair_indices_old = generate_batch_trials_ti(batch_items_old, num_train_trials, num_test_trials, arbitrary=args.arbitrary, mass_presentation=0)
    trials_old_revisited, correct_choices_old_revisited, pair_indices_old_revisited = generate_batch_trials_ti(batch_items_old, num_train_trials, num_test_trials, arbitrary=args.arbitrary, mass_presentation=0)
    trials_new, correct_choices_new, pair_indices_new = generate_batch_trials_ti(batch_items_new, num_train_trials, num_test_trials, arbitrary=args.arbitrary, mass_presentation=0)

    trials_old = torch.tensor(trials_old, dtype=torch.float32).to(device)
    trials_new = torch.tensor(trials_new, dtype=torch.float32).to(device)
    trials_old_test = torch.tensor(trials_old_revisited[:, num_train_trials:, :], dtype=torch.float32).to(device)
    trials = torch.cat([trials_old, trials_new, trials_old_test], dim=1).to(device)

    correct_choices_old = torch.tensor(correct_choices_old, dtype=torch.float32).to(device)
    correct_choices_new = torch.tensor(correct_choices_new, dtype=torch.float32).to(device)
    correct_choices_old_test = torch.tensor(correct_choices_old_revisited[:, num_train_trials:], dtype=torch.float32).to(device)
    correct_choices = torch.cat([correct_choices_old, correct_choices_new, correct_choices_old_test], dim=-1)

    correct_train_choices_old = 0
    correct_test_choices_old = 0
    correct_train_choices_new = 0
    correct_test_choices_new = 0
    correct_test_choices_old_revisited = 0

    plot_accuracies = {
        "old_train": 0.0,
        "old_test": 0.0,
        "new_train": 0.0,
        "new_test": 0.0,
        "old_revisited_test": 0.0,
    }

    accuracies_dict = {}

    total_trials = 2*(num_train_trials + num_test_trials) + num_test_trials

    for trial in range(total_trials):
        batch_trial = trials[:, trial, :]
        batch_correct_choice = correct_choices[:, trial]
        
        trial_input = batch_trial

        with torch.inference_mode():
            choice, neuromodulator, value, plastic_weights, hidden = model(trial_input, plastic_weights, batch_correct_choice)

        if torch.isnan(choice).any() or (choice < 0).any() or (choice > 1).any():
            print(f"Trial {trial}: choice has invalid values - min={choice.min()}, max={choice.max()}, nan={torch.isnan(choice).sum()}")
            break

        choice_sampled = torch.bernoulli(choice).squeeze(-1)

        if trial < num_train_trials:
            correct_train_choices_old += (choice_sampled == batch_correct_choice).sum().item()
        elif trial < num_train_trials + num_test_trials:
            correct_test_choices_old += (choice_sampled == batch_correct_choice).sum().item()
        elif trial < 2*num_train_trials + num_test_trials:
            correct_train_choices_new += (choice_sampled == batch_correct_choice).sum().item()
        elif trial < 2*num_train_trials + 2*num_test_trials:
            correct_test_choices_new += (choice_sampled == batch_correct_choice).sum().item()
        else:
            correct_test_choices_old_revisited += (choice_sampled == batch_correct_choice).sum().item()
    
    train_accuracy_old = correct_train_choices_old / (num_train_trials * batch_size)
    test_accuracy_old = correct_test_choices_old / (num_test_trials * batch_size)
    train_accuracy_new = correct_train_choices_new / (num_train_trials * batch_size)
    test_accuracy_new = correct_test_choices_new / (num_test_trials * batch_size)
    test_accuracy_old_revisited = correct_test_choices_old_revisited / (num_test_trials * batch_size)
    plot_accuracies["old_train"] = train_accuracy_old
    plot_accuracies["old_test"] = test_accuracy_old
    plot_accuracies["new_train"] = train_accuracy_new
    plot_accuracies["new_test"] = test_accuracy_new
    plot_accuracies["old_revisited_test"] = test_accuracy_old_revisited

    accuracies_dict.update({
        "old_train": train_accuracy_old,
        "old_test": test_accuracy_old,
        "new_train": train_accuracy_new,
        "new_test": test_accuracy_new,
        "old_revisited_test": test_accuracy_old_revisited,
    })

    plt.figure(dpi=300)
    plt.bar(list(plot_accuracies.keys()), list(plot_accuracies.values()))
    plt.xlabel('Trial Type')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    fig = plt.gcf()
    plt.close()
    model.train()
    
    return accuracies_dict, fig