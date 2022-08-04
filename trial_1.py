import numpy as np
import pickle
import matplotlib.pyplot as plt

BOARD = 10

class State:
    def __init__(self, p1, p2):
        self.board = np.zeros(BOARD)
        self.interval = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        # history of winnings
        self.history = []
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        # initialize p1 & p2 positions
        self.p1_position = 0
        self.p2_position = 0
        self.p1_shoot = False
        self.p2_shoot = False
        # self.boardHash = None
        # # init p1 plays first
        self.playerSymbol = 1
    
    def getHash(self):
        self.boardHash = str(self.board.reshape(BOARD))
        return self.boardHash

    def winner(self):
        # print("CALCULATE WINNING")
        # if P1 shoot
        # print("P1 shoot")
        if self.p1_shoot == True:
            t = self.p1_position
            winning_prob = np.random.uniform(0, 1)
            # Player 1 wins
            if winning_prob <= t:
                self.isEnd = True
                return 1
            # Player 2 wins
            else:
                self.isEnd = True
                return -1
        # if P2 shoot
        # print("P2 shoot")
        if self.p2_shoot == True:
            t = self.p2_position
            winning_prob = np.random.uniform(0, 1)
            # Player 2 wins
            if winning_prob <= t:
                self.isEnd = True
                return -1
            # Player 1 wins
            else:
                self.isEnd = True
                return 1
        # tie
        if self.p1_shoot == True and self.p2_shoot == True and self.p1_position == self.p2_position:
            self.isEnd = True
            return 0
        
        if self.p1_position == 1 or self.p2_position == 1:
            self.isEnd = True
            return 0
        
        # not end
        self.isEnd = False
        return None
    
    # get available positions
    def availablePositions(self):
        positions = []
        current_position = self.p2_position
        # append anything that is larger than current position to positions
        for i in self.interval:
            if i > current_position:
                positions.append(i)
        return positions

    # move the player to the next position
    def updateState(self, player):
        if player == "p2":
            if self.p2_position > 1.0 or self.p2_position > 1.0:
                self.isEnd = True
                return
            else:
                self.p1_position += 0.1
                self.p2_position += 0.1
        # switch to another player
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    # only when game ends
    def giveReward(self):
        result = self.winner()
        # backpropagate reward only to p2
        if result == 1:
            self.p2.feedReward(0)
        elif result == -1:
            self.p2.feedReward(1)
        else:
            self.p2.feedReward(0.5)

    # board reset
    def reset(self):
        self.board = BOARD
        self.isEnd = False
        self.p1_position = 0
        self.p2_position = 0
        self.p1_shoot = False
        self.p2_shoot = False
        self.playerSymbol = 1

    def play(self, rounds=100):
        for i in range(rounds):
            print("Rounds {}".format(i))
            shooting_pos_p1 = self.p1.chooseAction_p1() / 10
            shooting_pos_p2 = self.p2.chooseAction_p2_dummy() / 10
            while not self.isEnd:
                # Player 1
                # print("PLAYER 1")
                # position, round to 1 decimal point
                positions = round(self.p1_position, 1)
                # print("player 1 position: ", self.p1_position)
                if positions > 1:
                    print("Exiting")
                    break
                # print("current shooting stand: ", self.p1_shoot)
                if positions == shooting_pos_p1:
                    print("Player 1 shoot at ", positions)
                    self.p1_shoot = True
                    win = self.winner()
                else:
                    self.p1_shoot = False
                    # print("p1 shoot", self.p1_shoot)
                    # update board state
                    self.updateState("p1")
                    # check board status if it is end
                    win = self.winner()

                if win is not None:
                    # self.showBoard()
                    # ended with p1 either win or draw
                    # self.giveReward()
                    if win == 1:
                        self.history.append(1)
                    elif win == -1:
                        self.history.append(-1)
                    else:
                        self.history.append(0)
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    print("GAME ENDS - PLAYER 1 SHOT")
                    break

                else:
                    # print("PLAYER 2")
                    # Player 2
                    positions = round(self.p2_position, 1)
                    if positions == shooting_pos_p2:
                        print("Player 2 shoot at ", positions)
                        self.p2_shoot = True
                        win = self.winner()
                    else: 
                        self.p2_shoot = False
                        # update board state
                        self.updateState("p2")
                        # check board status if it is end
                        win = self.winner()
            
                    if win is not None:
                        # self.showBoard()
                        # ended with p2 either win or draw
                        self.giveReward()
                        if win == 1:
                            self.history.append(1)
                        elif win == -1:
                            self.history.append(-1)
                        else:
                            self.history.append(0)
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        print("GAME ENDS - PLAYER 2 SHOT")
                        break
            # reset board after each loop
            self.reset()


class Player:
    def __init__(self, name, exp_rate=0.3):
        self.name = name
        self.states = []  # record all positions taken
        self.lr = 0.2
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}  # state -> value

    def getHash(self, board):
        boardHash = str(board.reshape(BOARD))
        return boardHash
    
    # p1_action choose based on cdf 
    def chooseAction_p1(self):
        cdf_player1= [(0, 0.0),(0.0, 0.0), (0.0, 0.0), (0.0, 0.3437500000000011), (0.3437500000000011, 0.625000000000001), (0.625000000000001, 0.7777777777777787), (0.7777777777777787, 0.8698979591836744), 
        (0.8698979591836744, 0.929687500000001), (0.929687500000001, 0.97067901234568), (0.97067901234568, 1)]
        shoot_prob = np.random.uniform(0, 1)
        shooting_interval = 0
        # find which interval shoot_prob belongs to
        for i in cdf_player1:
            if i[0] <= shoot_prob <= i[1]:
                shooting_interval = i
        # find the index of the shooting interval in cdf_player1
        shooting_interval_index = cdf_player1.index(shooting_interval)
        # print("Player 1 shound shoot at: ", shooting_interval_index)
        return shooting_interval_index

    def chooseAction_p2_dummy(self):
        action = False
        cdf_player2= [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9], [0.9, 1]]
        shoot_prob = np.random.uniform(0, 1)
        shooting_interval = 0
        # find which interval shoot_prob belongs to
        for i in cdf_player2:
            if i[0] <= shoot_prob <= i[1]:
                shooting_interval = i
        # find the index of the shooting interval in cdf_player1
        shooting_interval_index = cdf_player2.index(shooting_interval)
        # print("Player 2 shound shoot at: ", shooting_interval_index)
        return shooting_interval_index

    # p2_action based on reinformcenet learning
    def chooseAction_p2(self, positions, current_board):
        if np.random.uniform(0, 1) <= self.exp_rate:
            # take random action
            return np.random.choice([True, False])
        else:
            value_max = -999
            for p in positions:
                next_board = positions.copy()
                # shoot
                # print(p)
                if p == 0:
                    p = 0
                else:
                    p = round(p/0.1)
                # print(p)
                next_board[p] = -1
                print(next_board)
                print(next_board[p])
                next_boardHash = str(next_board)
                value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash)
                # print("value", value)
                if value >= value_max:
                    value_max = value
                    shoot_position = p
        print("Player 2 will shoot at ", shoot_position)
        return shoot_position

    # append a hash state
    def addState(self, state):
        self.states.append(state)

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
            reward = self.states_value[st]

    def reset(self):
        self.states = []

    def savePolicy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()



if __name__ == "__main__":
    # training
    p1 = Player("p1")
    p2 = Player("p2")

    st = State(p1, p2)
    print("training...")
    st.play(50000)

    # get State history and plot
    history = st.history
    y = np.array(history)
    plt.hist(y)
    plt.show()
