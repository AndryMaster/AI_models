import random
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


def progress_bar(iteration, total, prefix='Progress:', suffix='', decimals=1, length=50, fill='█', lost='-'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + lost * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    if iteration == total:
        print()


def get_prices(share_symbol, start_date, end_date, cache_filename="prices.npy"):
    try:
        prices = np.load(cache_filename)
    except IOError:
        prices = yf.download(share_symbol, start_date, end_date)
        print(prices.keys())
        prices = prices['Open']
        np.save(cache_filename, prices)

    return prices.astype(float)


def plot_prices(prices):
    plt.title(f"Public stock price of {stock_symbol}")
    plt.xlabel("Day")
    plt.ylabel("Price ($)")
    plt.plot(prices)
    plt.savefig('prices.png')
    plt.show()


class DecisionPolicy:
    def select_action(self, current_state, step):
        pass

    def update_q(self, state, action, reward, next_state):
        pass


class RandomDecisionPolicy(DecisionPolicy):
    def __init__(self, actions):
        self.actions = actions

    def select_action(self, current_state, step):
        return random.choice(self.actions)


class QLearningDecisionPolicy(DecisionPolicy):
    def __init__(self, actions, input_dim):
        self.learning_rate = 0.01
        self.epsilon = 0.9  # 0.95
        self.gamma = 0.2  # 0.3
        self.actions = actions
        self.output_dim = len(actions)
        hidden_dim = 30  # 20

        self.x = tf.placeholder(tf.float32, shape=(None, input_dim))
        self.y = tf.placeholder(tf.float32, shape=(self.output_dim,))
        W1 = tf.Variable(tf.random_normal((input_dim, hidden_dim)))
        B1 = tf.Variable([0.1] * hidden_dim, shape=(hidden_dim,))
        H1 = tf.nn.relu(tf.matmul(self.x, W1) + B1)
        W2 = tf.Variable(tf.random_normal((hidden_dim, self.output_dim)))
        B2 = tf.Variable([0.1] * self.output_dim, shape=(self.output_dim,))
        print(W1.shape, B1.shape, H1.shape, W2.shape, B2.shape)
        self.q = tf.nn.relu(tf.matmul(H1, W2) + B2)

        loss = tf.square(self.y - self.q)
        self.train_op = tf.train.AdagradOptimizer(self.learning_rate).minimize(loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def select_action(self, current_state, step):
        threshold = min((self.epsilon, step / 1000.))
        if random.random() < threshold:
            # Exploit best option with probability epsilon
            action_q_vals = self.sess.run(self.q, feed_dict={self.x: current_state})
            action_idx = np.argmax(action_q_vals)  # TODO: replace w/ tensorflow's argmax
            action = self.actions[action_idx]
        else:
            # Explore random option with probability 1 - epsilon
            action = self.actions[random.randint(0, self.output_dim - 1)]
        return action

    def update_q(self, state, action, reward, next_state):
        action_q_vals = self.sess.run(self.q, feed_dict={self.x: state})
        next_action_q_vals = self.sess.run(self.q, feed_dict={self.x: next_state})
        current_action_idx = np.argmax(action_q_vals)
        next_action_idx = np.argmax(next_action_q_vals)
        # print(action_q_vals, current_action_idx)
        action_q_vals[0, current_action_idx] = reward + self.gamma * next_action_q_vals[0, next_action_idx]
        action_q_vals = np.squeeze(np.asarray(action_q_vals))
        # print(action_q_vals)
        # time.sleep(0.5)
        self.sess.run(self.train_op, feed_dict={self.x: state, self.y: action_q_vals})


def run_simulation(policy: DecisionPolicy, initial_budget, initial_num_stocks, prices, hist):
    budget, num_stocks = initial_budget, initial_num_stocks
    share_value = 0
    transitions = []
    max_iter = len(prices) - hist - 1
    for i in range(max_iter):
        if i == max_iter - 1:
            progress_bar(1, 1, suffix='Done')
        elif i % 70 == 0:
            progress_bar(i, max_iter, suffix=f'\tMoney: ${budget :.0f} Stocks: {num_stocks}  CurPrice: {prices[i]:.1f}')

        current_state = np.asmatrix(np.hstack((prices[i:i+hist], budget, num_stocks)))
        current_portfolio = budget + num_stocks * share_value
        action = policy.select_action(current_state, i)
        share_value = float(prices[i+hist])

        if action == 'Buy' and budget >= share_value:
            budget -= share_value
            num_stocks += 1
        elif action == 'Sell' and num_stocks > 0:
            budget += share_value
            num_stocks -= 1
        else:
            action = 'Hold'

        new_portfolio = budget + num_stocks * share_value
        reward = new_portfolio - current_portfolio
        next_state = np.asmatrix(np.hstack((prices[i+1:i+hist+1], budget, num_stocks)))
        transitions.append((current_state, action, reward, next_state))
        policy.update_q(*transitions[-1])

    portfolio = budget + num_stocks * share_value
    return portfolio


def run_simulations(policy: DecisionPolicy, initial_budget, initial_num_stocks, prices, hist):
    trains = 10
    final_portfolios = []
    for i in range(trains):
        portfolio = run_simulation(policy, initial_budget, initial_num_stocks, prices, hist)
        final_portfolios.append(portfolio / initial_budget)
        print(f"{i+1} Final portfolio: ${portfolio :.2f}")
    plt.title("Final Portfolio Value")
    plt.xlabel("Simulation №")
    plt.ylabel("Net worth")
    plt.plot(final_portfolios)
    plt.show()


if __name__ == "__main__":
    stock_symbol = 'QCOM'  # MSFT, AMD, NVDA, QCOM
    stocks_prices = get_prices(stock_symbol, '2000-06-22', '2020-06-22')
    print(f"Data length: {len(stocks_prices)}")
    plot_prices(stocks_prices)

    actions = ['Buy', 'Sell', 'Hold']
    bot_hist = 150
    # bot_policy = RandomDecisionPolicy(actions) 3
    bot_policy = QLearningDecisionPolicy(actions, bot_hist + 2)
    bot_budget = 10000.0
    bot_num_stocks = 0
    print(f"Start portfolio: ${bot_budget :.0f}")
    run_simulations(bot_policy, bot_budget, bot_num_stocks, stocks_prices, bot_hist)
