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

    model = QTable([3, 90, 90], [-3, -90, -90], 6, action_num)

    agents = [None]*agent_num
    for i in range(agent_num):
        agents[i] = QlearningAgent(action_num, model)

    actions = np.zeros((agent_num), dtype=int)

    continuous_count = np.zeros((agent_num), dtype=int)
    max_continuous_count = 0
    episode = 0

    loss_log = []
    max_continuous_log = []
    count_log = []
    continuous_log = [None]*agent_num

    fig, axis1 = plot.subplots()
    axis2 = axis1.twinx()
    axis1.legend(loc='lower right')
    axis2.legend(loc='lower right')

    for step in range(maxStep):
        for idx in range(agent_num):
            state = vector_observations[idx]
            reward = rewards[idx]
            is_end = done[idx]
            if is_end:
                agents[idx].stop(state, reward)
                actions[idx] = np.random.randint(2)
                continuous_count[idx] = 0
            else:
                actions[idx] = agents[idx].act_and_train(state, reward, episode)
                continuous_count[idx] += 1
                episode += 1

            if continuous_count[idx]>max_continuous_count:
                max_continuous_count = continuous_count[idx]

            if continuous_log[idx] is None:
                continuous_log[idx] = []
            continuous_log[idx].append(continuous_count[idx])

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

        count_log.append(step)
        max_continuous_log.append(max_continuous_count)
        loss_log.append(agents[0].get_statistics()["average_loss"])

        if step % 500 == 0:
            axis1.plot(count_log, max_continuous_log, label="max continuous count", color='r')
            axis2.plot(count_log, loss_log, label="average loss", color="c")

            continuous_log_colors = ["b","g","m","y","k","magenta"]
            for idx, val in enumerate(continuous_log):
                axis1.plot(count_log, val, label="continuous count [{}]".format(idx), color=continuous_log_colors[idx])

            plot.pause(0.001)
            plot.cla()




if __name__ == "__main__":

    env = UnityEnvironment(file_name=env_path)

    num = 6

    #random_test(env, num, 10000)
    q_learning(env, num, 10**5)

    env.close()