import numpy as np
import matplotlib.pyplot as plot

from unityagents import UnityEnvironment

env_path = 'CartPole_DQN'
brain_name = 'CartPoleBrain'

def random_test(env, num, maxStep):

    action_info = env.reset(train_mode=True)
    for step in range(maxStep):

        action = np.random.randint(0, 2, (num))

        action_info = env.step(vector_action=action)

        state = action_info[brain_name].vector_observations
        image = action_info[brain_name].visual_observations

        reward = action_info[brain_name].rewards

        done = action_info[brain_name].local_done

        max_reach = action_info[brain_name].max_reached

        print("========={} step =========".format(step))
        print("action=", action)
        print("state =", state)
        print("image =", image)
        print("done  =", done)
        print("max_reach", max_reach)

def q_learning(env, num, maxStep):

    from QlearningAgents import QlearningAgent, QTable

    action_info = env.reset(train_mode=True)
    brain_agents = action_info[brain_name].agents
    vector_observations = action_info[brain_name].vector_observations
    done = action_info[brain_name].local_done
    rewards = action_info[brain_name].rewards

    agent_num = len(brain_agents)
    action_num = 2
    max_step_size = 500.0

    model = QTable([3, 90, 90], [-3, -90, -90], 8, action_num)

    agents = [None]*agent_num
    for i in range(agent_num):
        agents[i] = QlearningAgent(action_num, model)

    actions = np.zeros((agent_num), dtype=int)

    continuous_count = np.zeros((agent_num), dtype=int)
    loss_log = []
    epsilon_log = []
    continuous_rate_log = [[]]*agent_num
    episode_log = [[]]*agent_num
    step_log = []

    for step in range(maxStep):
        for idx in range(agent_num):
            state = vector_observations[idx]
            reward = rewards[idx]
            is_end = done[idx]

            if is_end:
                agents[idx].stop(state, reward)
                actions[idx] = np.random.randint(2)

                #for log
                episode_log[idx] = episode_log[idx] + [len(episode_log[idx])+1]
                continuous_rate_log[idx] = continuous_rate_log[idx] + [continuous_count[idx]*100.0/max_step_size]
                continuous_count[idx] = 0
            else:
                actions[idx] = agents[idx].act_and_train(state, reward, step)

                #for log
                continuous_count[idx] += 1

        action_info = env.step(vector_action=actions)

        vector_observations = action_info[brain_name].vector_observations
        rewards = action_info[brain_name].rewards
        done = action_info[brain_name].local_done

        max_reach = action_info[brain_name].max_reached

        """
        print("========={} step =========".format(step))
        print("action=", actions)
        print("state =", vector_observations)
        print("done  =", done)
        print("max_reach =", max_reach)
        print("max_continuous_count =", max_continuous_count)
        print("statistic", agents[0].get_statistics())
        """

        step_log.append(step)
        loss_log.append(agents[0].get_statistics()["average_loss"])
        epsilon_log.append(agents[0].get_statistics()["epsilon"])

        #if step % 500 == 0:


    plot.subplot(3, 1, 1)
    plot.plot(step_log, loss_log, label="average loss")
    plot.legend(loc='lower right', fontsize=8)

    plot.subplot(3, 1, 2)
    plot.plot(step_log, epsilon_log, label="epsilon")
    plot.legend(loc='lower right', fontsize=8)

    plot.subplot(3,1,3)
    continuous_log_colors = ["b","g","m","y","k","magenta"]
    for idx, val in enumerate(continuous_rate_log):
        plot.plot(episode_log[idx], val, label="continuous count [{}]".format(idx), color=continuous_log_colors[idx])
    plot.legend(loc='lower right', fontsize=8)

    """
    fig, axis1 = plot.subplots()
    axis2 = axis1.twinx()
    axis1.legend(loc='lower right')
    axis2.legend(loc='lower right')

    max_episode_log = max(episode_log, key=len)
    axis2.plot(max_episode_log, loss_log, label="average loss", color="c")

    continuous_log_colors = ["b","g","m","y","k","magenta"]
    for idx, val in enumerate(continuous_rate_log):
        axis1.plot(episode_log[idx], val, label="continuous count [{}]".format(idx), color=continuous_log_colors[idx])
    """

    plot.show()




if __name__ == "__main__":

    env = UnityEnvironment(file_name=env_path)

    num = 6

    #random_test(env, num, 10000)
    q_learning(env, num, 10**4)

    env.close()