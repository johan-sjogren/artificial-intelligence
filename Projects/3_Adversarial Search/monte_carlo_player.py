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


class MCTSPlayer(DataPlayer):
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
