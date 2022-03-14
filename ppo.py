import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import game, trainAgent, randomAgent, neuralAgent
import numpy as np
from torch.optim import Adam
import time
import network, network2
import card
import torch.nn.functional as F

class PPO:
    def __init__(self):
        """
            Initializes the PPO model, including hyperparameters.
            Parameters:
                policy_class - the policy class to use for our actor/critic networks.
                env - the environment to train on.
                hyperparameters - all extra arguments passed into PPO that should be hyperparameters.
            Returns:
                None
        """

        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters()
        # Initialize actor and critic networks
        self.actor = network.FeedForwardNN(121, 55)  # ALG STEP 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic = network2.FeedForwardNN(121, 1)
        try:
            self.actor.load_state_dict(torch.load('./ppo_actor.pth'))
            self.critic.load_state_dict(torch.load('./ppo_critic.pth'))
        except:
            pass

        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,  # timesteps so far
            'i_so_far': 0,  # iterations so far
            'batch_lens': [],  # episodic lengths in batch
            'batch_rews': [],  # episodic returns in batch
            'actor_losses': [],  # losses of actor network in current iteration
            'win_rate': []
        }

    def _init_hyperparameters(self):
        """
            Initialize default and custom values for hyperparameters
            Parameters:
                hyperparameters - the extra arguments included when creating the PPO model, should only include
                                    hyperparameters defined below with custom values.
            Return:
                None
        """
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.timesteps_per_batch = 50000  # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 200000  # Max number of timesteps per episode
        self.current_timesteps_per_episode = self.max_timesteps_per_episode
        self.n_updates_per_iteration = 10  # Number of times to update actor/critic per iteration
        self.lr = 0.002  # Learning rate of actor optimizer
        self.gamma = 0.97     # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2

        # Miscellaneous parameters
        self.render = True  # If we should render during rollout
        self.render_every_i = 10  # Only render every n iterations
        self.save_freq = 1  # How often we save in number of iterations

    def compute_rtgs(self, batch_rews):
        """
            Compute the Reward-To-Go of each timestep in a batch given the rewards.
            Parameters:
                batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)
            Return:
                batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
        """
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0  # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * 0.9
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    # generates the data for the AI to train on
    def generate_data(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []
        amountOfWins = 0
        batch_masks = []
        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []

        t = 0  # Keeps track of how many timesteps we've run so far this batch

        AI = trainAgent.Agent(None)
        self.actor.to(torch.device("cpu"))
        self.actor.eval()
        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < self.current_timesteps_per_episode:
            ep_rews = []  # rewards collected per episode
            _game = game.Game(4)
            _game.players.append(trainAgent.Agent(_game))
            _game.players.append(trainAgent.Agent(_game))
            _game.players.append(trainAgent.Agent(_game))
            _game.players.append(trainAgent.Agent(_game))
            _game.players[0].nn = self.actor
            _game.players[1].nn = self.actor
            _game.players[2].nn = self.actor
            _game.players[3].nn = self.actor
            result = _game.auto_sim()
            single_exp = False
            if not single_exp:
                for x in range(len(_game.players)):
                    ep_obs = _game.players[x].episode_obs
                    episode_len = len(ep_obs)
                    t += episode_len
                    if episode_len > 0:
                        batch_lens.append(episode_len)
                        rewards = np.zeros(shape=[episode_len])
                        batch_obs.extend(ep_obs)
                        batch_acts.extend(_game.players[x].episode_act)
                        batch_log_probs.extend(_game.players[x].episode_logprobs)
                        batch_masks.extend(_game.players[x].episode_mask)
                        if result[1] is None:
                            rewards[-1] = -0.15 - 0.01 * episode_len
                        elif result[1] == x:
                            rewards[-1] = 1 - 0.01 * episode_len
                        else:
                            rewards[-1] = -1/3 - 1/3 * 0.01 * episode_len
                        batch_rews.append(rewards)
            else:
                x = np.random.randint(0, 3)
                ep_obs = _game.players[x].episode_obs
                episode_len = len(ep_obs)
                t += episode_len
                if episode_len > 0:
                    batch_lens.append(episode_len)
                    rewards = np.zeros(shape=[episode_len])
                    batch_obs.extend(ep_obs)
                    batch_acts.extend(_game.players[x].episode_act)
                    batch_log_probs.extend(_game.players[x].episode_logprobs)
                    batch_masks.extend(_game.players[x].episode_mask)
                    if result[1] is None:
                        rewards[-1] = -0.2
                    elif result[1] == x:
                        rewards[-1] = 1.5
                    else:
                        rewards[-1] = -1 / 3
                    batch_rews.append(rewards)
        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)  # ALG STEP 4
        batch_masks = torch.tensor(batch_masks, dtype=torch.int)
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens
        self.actor.to(torch.device(self.device))
        self.actor.train()
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_masks

    # TODO create test function to test AI performance
    def test(self):
        # simulate game against random agents and return wins and losses and draws
        wins = 0
        losses = 0
        draws = 0
        self.actor.to(torch.device("cpu"))
        self.actor.eval()
        for x in range(10000):
            ind = np.random.randint(0, 3)
            _game = game.Game(4)
            for y in range(ind):
                _game.players.append(randomAgent.Agent())
            _game.players.append(neuralAgent.Agent(_game))  # TODO write performance agent and use it
            _game.players[ind].nn = self.actor
            for z in range(3 - ind):
                _game.players.append(randomAgent.Agent())
            result = _game.auto_sim()
            if result[1] is None:
                draws += 1
            elif result[1] == 0:
                wins += 1
            else:
                losses += 1
        return wins, losses, draws


    def evaluate(self, batch_obs, batch_acts, batch_masks):
        """
            Estimate the values of each observation, and the log probs of
            each action in the most recent batch with the most recent
            iteration of the actor network. Should be called from learn.
            Parameters:
                batch_obs - the observations from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of observation)
                batch_acts - the actions from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of action)
            Return:
                V - the predicted values of batch_obs
                log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
        """
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V = self.critic(batch_obs.to(torch.device("cpu"))).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs.to(self.device)).to(torch.device("cpu"))
        mean = mean * batch_masks
        mean2 = mean.clone()
        #torch.cuda.synchronize()
        dist = Categorical(mean)
        log_probs = dist.log_prob(batch_acts)
        for mea in mean:
            if torch.count_nonzero(mea) == 0:
                print(mea)
        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
       #torch.cuda.synchronize()
        return V, log_probs, dist.entropy()

    def learn(self):
        """
            Train the actor and critic networks. Here is where the main PPO algorithm resides.
            Parameters:
                total_timesteps - the total number of timesteps to train for
            Return:
                None
        """
        print(f"Learning... Running {40} timesteps per episode, ", end='')
        print(f"{10000} timesteps per batch for a total of {20000} timesteps")
        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations ran so far
        while True:  # ALG STEP 2
            # Autobots, roll out (just kidding, we're collecting our batch simulations here)
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_masks = self.generate_data()  # ALG STEP 3
            self.current_timesteps_per_episode = self.max_timesteps_per_episode
            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # Calculate advantage at k-th iteration
            V, _ = self.evaluate(batch_obs, batch_acts, batch_masks)[0:2]
            A_k = batch_rtgs - V.detach()  # ALG STEP 5

            # One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
            # isn't theoretically necessary, but in practice it decreases the variance of
            # our advantages and makes convergence much more stable and faster. I added this because
            # solving some environments was too unstable without it.
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # This is the loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):  # ALG STEP 6 & 7
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs, entropy = self.evaluate(batch_obs, batch_acts, batch_masks)
                # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                # NOTE: we just subtract the logs, which is the same as
                # dividing the values and then canceling the log with e^log.
                # For why we use log probabilities instead of actual probabilities,
                # here's a great explanation:
                # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
                # TL;DR makes gradient ascent easier behind the scenes.
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                ratio = torch.mean(torch.pow(ratios, 2) - 1).item()
                print("Ratio: " + str(ratio))
                #if ratio > 0.08:
                #    break
                # Calculate surrogate losses.
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Calculate actor and critic losses.
                # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                # the performance function, but Adam minimizes the loss. So minimizing the negative
                # performance function maximizes it.
                actor_loss = (-torch.min(surr1, surr2)).mean() - 0.02 * entropy.mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Log actor loss
                self.logger['actor_losses'].append(actor_loss.detach())

            # Print a summary of our training so far
            self._log_summary()

            # Save our model if it's time
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')
            if i_so_far % 10 == 0:
                print(self.test())

    def _log_summary(self):
        """
            Print to stdout what we've logged so far in the most recent batch.
            Parameters:
                None
            Return:
                None
        """
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 3))
        avg_actor_loss = str(round(avg_actor_loss, 5))
        win_rate = self.logger['win_rate']

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"Iteration took: {win_rate} %", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []
