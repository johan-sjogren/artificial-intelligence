import random
import math
from sample_players import DataPlayer


def random_roll_out(state):
    return random.choice(state.actions())

class Node:

    def __init__(self, state, parent, player_id):
        self.parent = parent
        self.state = state
        self.untested_actions = state.actions()
        self.children = [] 
        self.actions = []
        self.tot_plays = 0
        self.wins = 0
        self.player_id = player_id
    
    def __str__(self):
        return """Node(
            parent={}
            state={}
            children={}
            actions={}
            tot_plays={}
            total_wins={}
            player_id={}
        )""".format(self.parent, self.state, self.children, self.actions, self.tot_plays, self.wins, self.player_id)

    def is_terminal(self):
        return self.state.terminal_test()

    def is_expanded(self):
        return len(self.untested_actions) == 0
    
    def expand(self):
        action = self.untested_actions.pop()
        child_state = self.state.result(action)
        child_node = Node(child_state, self, self.player_id)
        self.children.append(child_node)
        self.actions.append(action)
        return child_node

    def roll_out(self, roll_out_policy=random_roll_out):
        state = self.state
        while not state.terminal_test():
            action = roll_out_policy(state)
            state = state.result(action)
        reward = Node.convert_reward(state.utility(self.player_id))
        return reward, state.player()

    @staticmethod
    def convert_reward(reward):
        if reward == float("inf"):
            return 1
        elif reward == float("-inf"):
            return 0
        return reward

    def get_best_action(self):
        child_ucts = [child.UCT(const=0) for child in self.children]
        # child_ucts = [child.tot_plays for child in self.children]
        # child_ucts = [child.wins for child in self.children] # Bad
        if len(child_ucts) == 0:
            action = random.choice(self.untested_actions)
            child_state = self.state.result(action)
            child_node = Node(child_state, self, self.player_id)
            self.children.append(child_node)
            self.actions.append(action)
            return action, child_state

        arg_max = max(enumerate(child_ucts), key=lambda x: x[1])[0]
        return self.actions[arg_max], self.children[arg_max]

    def backpropagate(self, reward):
        self.wins += reward
        reward ^= 1
        self.tot_plays += 1
        if self.parent is not None:
            self.parent.backpropagate(reward)
    
    def UCT(self, const = math.sqrt(2)):
        if self.tot_plays > 0:
            uct = self.wins/self.tot_plays
        else:
            return 0.0
        if self.parent is not None:
            uct += const * math.sqrt(2*math.log(self.parent.tot_plays)/self.tot_plays)
        return uct

    def select(self):
        child_ucts = [child.UCT()for child in self.children]
        arg_max = max(enumerate(child_ucts), key=lambda x: x[1])[0]
        return self.children[arg_max]


class CustomPlayer(DataPlayer):
    def get_action(self, state):
        self.run_mcts(state)

    def run_mcts(self, root_state):
        """ Monte Carlo Tree Search has 4 steps:
            1. Selection
            2. Expansion
            3. Roll out
            4. Back propagation 
        """
        def select_node(node):
            while not node.is_terminal():
                if not node.is_expanded():
                    return node.expand()
                else:
                    node = node.select()
            return node

        steps = 0
        root_node = Node(root_state, None, self.player_id)

        if isinstance(self.context, Node):
            prev_child_node = self.context
            for prev in prev_child_node.children:
                if prev.state == root_state:
                    root_node = prev 
                    break

        while True:
            leaf_node = select_node(root_node) # Expansion happens here as well
            result, final_player = leaf_node.roll_out() #Simulate
            if final_player == self.player_id:
                result ^= 1
            leaf_node.backpropagate(result)
            best_action, best_child_node  = root_node.get_best_action()

            self.queue.put(best_action)
            best_child_node.parent = None
            self.context = best_child_node 


class CustomPlayerBase(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        # Use minimax with my_open - oppenent_open heuristic
        # self._minimax_action(state, depth=3, heuristic=self.score)
        # Iterative deepening_action
        self._iterative_deepening_action(state, 10, heuristic=self.score)

    def min_value(self, state, depth, heuristic):
        if state.terminal_test(): return state.utility(self.player_id)
        if depth <= 0: return heuristic(state) # This is the heuristic
        value = float("inf")
        for action in state.actions():
            value = min(value, self.max_value(state.result(action), depth - 1, heuristic))
        return value

    def max_value(self, state, depth, heuristic):
        if state.terminal_test(): return state.utility(self.player_id)
        if depth <= 0: return heuristic(state)
        value = float("-inf")
        for action in state.actions():
            value = max(value, self.min_value(state.result(action), depth - 1, heuristic))
        return value

    def minimax(self, state, depth, heuristic):
        return max(state.actions(),
                   key=lambda x: self.min_value(state.result(x), depth - 1, heuristic))

    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)

    def _minimax_action(self, state, depth, heuristic):
        """ Choose an action available in the current state

        See RandomPlayer and GreedyPlayer for examples.

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller is responsible for
        cutting off the function after the search time limit has expired. 

        **********************************************************************
        NOTE: since the caller is responsible for cutting off search, calling
              get_action() from your own code will create an infinite loop!
              See (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # return the optimal minimax move at a fixed search depth 
        self.queue.put(self.minimax(state, depth, heuristic))

    def _iterative_deepening_action(self,state, depth_limit, heuristic):
        # Turns out "iterative deepening" is just a for loop...
        best_move = None
        for depth in range(1, depth_limit+1):
            best_move = self.minimax(state, depth, heuristic)
            self.queue.put(best_move)


class IterativeAlphaBeta(CustomPlayerBase):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        self._iterative_alphabeta_action(state, 20, heuristic=self.score)

    def _iterative_alphabeta_action(self,state, depth_limit, heuristic):
        # Turns out "iterative deepening" is just a for loop...
        best_move = None
        for depth in range(1, depth_limit+1):
            best_move = self.alpha_beta_search(state, depth, heuristic)
            self.queue.put(best_move)

    def alpha_beta_search(self, gameState, depth, heuristic):
        """ Return the move along a branch of the game tree that
        has the best possible value.  A move is a pair of coordinates
        in (column, row) order corresponding to a legal move for
        the searching player.
    
        You can ignore the special case of calling this function
        from a terminal state.
        """
        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None
        for a in gameState.actions():
            v = self.ab_min_value(gameState.result(a), alpha, beta, depth - 1, heuristic)
            alpha = max(alpha, v)
            if v > best_score:
                best_score = v
                best_move = a
        return best_move

    def ab_min_value(self, state, alpha, beta, depth, heuristic):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        if state.terminal_test(): return state.utility(self.player_id)
        if depth <= 0: return heuristic(state) # This is the heuristic
    
        v = float("inf")
        for a in state.actions():
            v = min(v, self.ab_max_value(state.result(a), alpha, beta, depth - 1, heuristic))
            if v <= alpha: return v
            beta = min(beta, v)
        return v

    def ab_max_value(self, state, alpha, beta, depth, heuristic):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        if state.terminal_test(): return state.utility(self.player_id)
        if depth <= 0: return heuristic(state) # This is the heuristic

        v = float("-inf")
        for a in state.actions():
            v = max(v, self.ab_min_value(state.result(a), alpha, beta, depth - 1, heuristic))
            if v >= beta: return v
            alpha = max(alpha, v)
        return v