import numpy as np

debug_view = False

class QTable:
    def __init__(self, maxs, mins, binNum, actionNum):
        assert len(maxs)==len(mins)
        self.state_dim = len(maxs)
        self.bin_num = binNum
        self.state_min = mins
        self.state_max = maxs
        self.q_table = np.random.rand(binNum**self.state_dim, actionNum)

    def __call__(self, state, idx=None):
        self._check_state(state)
        dim = self._digitize_state(state)

        if idx is None:
            return self.q_table[dim]
        else:
            assert idx < len(self.q_table[0])
            return [self.q_table[dim][idx]]

    def _digitize_state(self, state):
        digitized = [
            np.digitize(state[i], np.linspace(self.state_min[i], self.state_max[i], self.bin_num + 1)[1:-1]) for i
            in range(len(state))]
        return sum([x * (self.bin_num ** i) for i, x in enumerate(digitized)])

    def view(self):
        print(self.q_table)

    def _check_state(self, state):
        assert len(state)==self.state_dim
        assert np.any(state<self.state_min) == False, "min={},state={}".format(self.state_min, state)
        assert np.any(state>self.state_max) == False, "max={},state={}".format(self.state_max, state)

    def update(self, state, action, loss):
        dim = self._digitize_state(state)
        self.q_table[dim][action] += loss



class QlearningAgent:
    def __init__(self, actionNum, model, alpha=0.1, gamma=0.99):
        assert alpha > 0 and alpha <= 1
        assert gamma > 0 and gamma <= 1

        self.model = model
        self.action_num = actionNum
        self.pre_state = None
        self.curr_state = None
        self.reward = None
        self.gamma = gamma
        self.alpha = alpha
        self.pre_action = None
        self.loss = 0
        self.loss_gamma = 0.999
        self.epsilon = 0

    def _td_loss(self, reward, isLast=False):
        if isLast is True:
            td = reward - max(self.model(self.pre_state, self.pre_action))
        else:
            td = reward + self.gamma * max(self.model(self.curr_state)) - max(self.model(self.pre_state,self.pre_action))
        return td


    def _update(self, reward):

        is_last = self.curr_state == None
        loss = self._td_loss(reward, is_last)
        self.model.update(self.pre_state, self.pre_action, self.alpha*loss)
        if debug_view:
            self.model.view()

        self.loss = self.loss * self.loss_gamma + (1.0 - self.loss_gamma) * loss



    def act_and_train(self, state, reward, episode=0):
        if self.curr_state is not None:
            self.pre_state = self.curr_state
        self.curr_state = state

        if self.pre_state is not None and self.pre_action is not None:
            self._update(reward)

        self.epsilon = (1.0 / (episode + 1))
        if self.epsilon <= np.random.uniform(0, 1):
            if debug_view:
                print("call model")
            act = np.argmax(self.model(state))
        else:
            if debug_view:
                print("random action")
            act = np.random.randint(self.action_num)

        self.pre_action = act

        return act


    def stop(self, state, reward):
        if self.curr_state is not None:
            self.pre_state = self.curr_state
            self.curr_state = None
            self._update(reward)
        self.pre_action = None

    def _compute_loss(self):
        pass

    def act(self, state):
        act = np.argmax(self.model(state))

    def get_statistics(self):
        return {"average_loss":self.loss, "epsilon":self.epsilon}

if __name__ == "__main__":
    #agent = Qlearning([4,4,4], [0, 0, 0], 5, 2)
    model = QTable([3, 90, 90], [-3, -90, -90], 5, 2)
    agent = QlearningAgent(2, model)
    state = [0, 0, 0]
    act = agent.act_and_train(state, 1)
    act = agent.act_and_train(state, 1)
    print("act={}".format(act))