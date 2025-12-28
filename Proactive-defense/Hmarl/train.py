import torch
import numpy as np
from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent, B_lineAgent,SleepAgent,EnterpriseGreenAgent,FiniteStateRedAgent,MainAgent
from CybORG.Agents.SimpleAgents.DQNAgent import DQNAgent
from CybORG.Agents.Wrappers import ChallengeWrapper
from .utils import plot_learning_curve
import inspect
import os

from CybORG import CybORG
from CybORG.Simulator.Scenarios import ICSNetworkGenerator
from CybORG.Agents.Wrappers import EnterpriseMAE


PATH = str(inspect.getfile(CybORG))
PATH = PATH[:-10] + '/Shared/Scenarios/Scenario1b.yaml'


# check if cuda is available
def cuda():
    print("CUDA: " + str(torch.cuda.is_available()))


def train_DDQN(red_agent=RedMeanderAgent, num_eps=2200, len_eps=100, replace=1000, mem_size=5000,
lr=0.0001, eps_dec=0.000005, eps_min=0.05, gamma=0.99, batch_size=32, epsilon=1, chkpt_dir="model_meander"):
    sg = ICSNetworkGenerator(
        blue_agent_class=MainAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=red_agent,
        steps=500
    )
    cyborg = CybORG(scenario_generator=sg)
    env = EnterpriseMAE(env=cyborg)

    model_dir = os.path.join(os.getcwd(), "Models", chkpt_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    agent = DQNAgent(gamma=gamma, epsilon=epsilon, lr=lr,
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=mem_size, eps_min=eps_min,
                     batch_size=batch_size, replace=replace, eps_dec=eps_dec,
                     chkpt_dir=model_dir, algo='DDQNAgent',
                     env_name='Scenario1b')

    best_score = -np.inf
    # not using a checkpoint (i.e new model)
    load_checkpoint = False
    if load_checkpoint:
        agent.load_models()

    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(num_eps):
        score = 0
        # need to reset the environment at the end of the episode (this could also be done by using end_episode() of the red agent)
        observation = env.reset()
        for j in range(len_eps):
            action = agent.get_action(observation)
            observation_, reward, done, info = env.step(action=action)
            score += reward
            if not load_checkpoint:
                agent.store_transition(observation, action,
                                     reward, observation_, int(done))
                agent.train()
            observation = observation_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i,'score: ', score,
             ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
            'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

        # keep track of best score to see if we are converging
        if avg_score > best_score:
            best_score = avg_score

        eps_history.append(agent.epsilon)

    # plot learning curves (moving average over last 20 episodes and shows epsilon decreasing)
    # will generate png to the models directory
    # learning curves are misleading since epsilon = 0.05 means 5% chance of random action, it should be evaluated every 200 episodes without epsilon
    plot_learning_curve(steps_array, scores, eps_history, os.path.join(model_dir, "plot.png"))
    # save model so we can use it after training complete
    # it will be stored in model_dir (both models [i.e the target one also so we can train again if needed])
    agent.save_models()


# this is used to see how well the model performs (without epsilon)
# and to generate test cases to see what the model is doing
def test_DDQN(eps_len=100, num_eps=10, chkpt_dir="model_b_line", red_agent=B_lineAgent):
    sg = ICSNetworkGenerator(
            blue_agent_class=MainAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=red_agent,
            steps=500
        )
    cyborg = CybORG(scenario_generator=sg)
    env = EnterpriseMAE(env=cyborg)

    model_dir = os.path.join(os.getcwd(), "Models", chkpt_dir)
    # the default epsilon is 0. we also don't need to define most hyperparamters since all we will do is agent.get_action()
    agent = DQNAgent(chkpt_dir=model_dir, algo='DDQNAgent',
                     env_name='Scenario1b')
    # gets the checkpoint from model_dir
    agent.load_models()

    scores = []
    
    for i in range(num_eps):
        s = []
        a = []
        observation = env.reset()
        for j in range(eps_len):
            action = agent.get_action(observation)
            observation, reward, done, info = env.step(action=action)
            s.append(reward)
            a.append((str(env.get_last_action('Blue')), str(env.get_last_action('Red'))))
        total_score = np.sum(s)
        scores.append(total_score)
        print('score: ', total_score)
        print('actions: ', a)
    avg_score = np.mean(scores)
    print('average score: ', avg_score)


if __name__ == '__main__':
    cuda()
    # we should tune hyperparameters here (with random search)
    #train_DDQN(red_agent=RedMeanderAgent, num_eps=2000, len_eps=100, replace=5000, mem_size=5000, lr=0.0001, eps_dec=0.000005, eps_min=0.05, gamma=0.99, batch_size=32, epsilon=1, chkpt_dir="model_meander2")
    test_DDQN(chkpt_dir="model_b_line", red_agent=B_lineAgent)
