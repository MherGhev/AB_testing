import numpy as np
import matplotlib.pyplot as plt
import csv
from loguru import logger

class Bandit:
    """
    Abstract class representing a single Bandit.

    Attributes:
        p (float): The true probability of reward for the bandit.
        p_estimate (float): The estimated probability of reward.
        N (int): Number of times the bandit has been pulled.
    """
    def __init__(self, p):
        self.p = p  # true probability of winning
        self.p_estimate = 0.0  # estimated probability
        self.N = 0  # number of times this bandit was pulled

    def pull(self):
        """
        Simulates pulling the bandit's arm.

        Returns:
            int: 1 if the pull results in a reward, 0 otherwise.
        """
        return 1 if np.random.random() < self.p else 0

    def update(self, reward):
        """
        Updates the estimated probability of reward based on observed data.

        Args:
            reward (int): The observed reward (1 for success, 0 for failure).
        """

        self.N += 1
        self.p_estimate = ((self.N - 1) * self.p_estimate + reward) / self.N


class EpsilonGreedy(Bandit):
    """
    Implements the Epsilon-Greedy algorithm.

    Attributes:
        epsilon (float): Exploration probability.
    """
    def __init__(self, p, epsilon=0.1):
        super().__init__(p)
        self.epsilon = epsilon

    def experiment(self, bandits, trials=20000):
        """
        Runs the Epsilon-Greedy experiment.

        Args:
            bandits (list): List of bandits.
            trials (int): Number of trials to run.

        Returns:
            tuple: (list of rewards, list of regrets).
        """
        rewards = []
        regrets = []
        optimal_bandit = np.argmax([b.p for b in bandits])
        for t in range(1, trials + 1):
            self.epsilon = 1 / t  # decay epsilon
            if np.random.random() < self.epsilon:
                chosen_bandit = np.random.choice(len(bandits))
            else:
                chosen_bandit = np.argmax([b.p_estimate for b in bandits])

            reward = bandits[chosen_bandit].pull()
            bandits[chosen_bandit].update(reward)
            rewards.append(reward)
            regret = bandits[optimal_bandit].p - bandits[chosen_bandit].p
            regrets.append(regret)
        return rewards, regrets


class ThompsonSampling(Bandit):
    """
    Implements the Thompson Sampling algorithm.

    Attributes:
        a (int): Alpha parameter for the Beta distribution.
        b (int): Beta parameter for the Beta distribution.
    """
    def __init__(self, p):
        super().__init__(p)
        self.a = 1  # alpha
        self.b = 1  # beta

    def update(self, reward):
        """
        Updates the Beta distribution parameters based on observed reward.

        Args:
            reward (int): The observed reward (1 for success, 0 for failure).
        """
        self.a += reward
        self.b += 1 - reward

    def experiment(self, bandits, trials=20000):
        """
        Runs the Thompson Sampling experiment.

        Args:
            bandits (list): List of bandits.
            trials (int): Number of trials to run.

        Returns:
            tuple: (list of rewards, list of regrets).
        """
        rewards = []
        regrets = []
        optimal_bandit = np.argmax([b.p for b in bandits])
        for _ in range(trials):
            sampled_probs = [np.random.beta(b.a, b.b) for b in bandits]
            chosen_bandit = np.argmax(sampled_probs)

            reward = bandits[chosen_bandit].pull()
            bandits[chosen_bandit].update(reward)
            rewards.append(reward)
            regret = bandits[optimal_bandit].p - bandits[chosen_bandit].p
            regrets.append(regret)
        return rewards, regrets


def plot_learning(rewards, regrets, algorithm_name):
    """
    Plots cumulative rewards and regrets for a given algorithm.

    Args:
        rewards (list): List of rewards observed over trials.
        regrets (list): List of regrets observed over trials.
        algorithm_name (str): Name of the algorithm being visualized.
    """
    cumulative_rewards = np.cumsum(rewards)
    cumulative_regrets = np.cumsum(regrets)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(cumulative_rewards)
    plt.title(f'{algorithm_name} - Cumulative Rewards')
    plt.xlabel('Trial')
    plt.ylabel('Cumulative Reward')

    plt.subplot(1, 2, 2)
    plt.plot(cumulative_regrets)
    plt.title(f'{algorithm_name} - Cumulative Regrets')
    plt.xlabel('Trial')
    plt.ylabel('Cumulative Regret')

    plt.tight_layout()
    plt.show()


def save_to_csv(bandits, rewards, algorithm_name, filename="results.csv"):
    """
    Saves experiment results to a CSV file.

    Args:
        bandits (list): List of bandits.
        rewards (list): List of rewards observed during the experiment.
        algorithm_name (str): Name of the algorithm used.
        filename (str, optional): File name for the CSV file. Defaults to "results_corrected.csv".
    """
    data = [
        [i + 1, reward, algorithm_name]
        for i, reward in enumerate(rewards)
    ]  # Bandit number, reward, algorithm name

    print("data: ", data)
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Bandit", "Reward", "Algorithm"])
        writer.writerows(data)


bandit_probs = [0.1, 0.2, 0.3, 0.4]
num_trials = 20000

# Epsilon-Greedy Experiment
logger.info("Running Epsilon-Greedy Experiment")
eg_bandits = [EpsilonGreedy(p) for p in bandit_probs]
eg_rewards, eg_regrets = eg_bandits[0].experiment(eg_bandits, trials=num_trials)
plot_learning(eg_rewards, eg_regrets, "Epsilon-Greedy")

# Thompson Sampling Experiment
logger.info("Running Thompson Sampling Experiment")
ts_bandits = [ThompsonSampling(p) for p in bandit_probs]
ts_rewards, ts_regrets = ts_bandits[0].experiment(ts_bandits, trials=num_trials)
plot_learning(ts_rewards, ts_regrets, "Thompson Sampling")


results = [("Bandit", "Reward", "Epsilon-Greedy")] * len(eg_rewards) + \
          [("Bandit", "Reward", "Thompson Sampling")] * len(ts_rewards)

logger.info("Saving Epsilon-Greedy results to CSV")
save_to_csv(eg_bandits, eg_rewards, "Epsilon-Greedy")


logger.info("Saving Thompson Sampling results to CSV")
save_to_csv(ts_bandits, ts_rewards, "Thompson Sampling")