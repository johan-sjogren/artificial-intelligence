
from sample_players import DataPlayer
import random

class CustomPlayer(DataPlayer):
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
        # self._minimax_alphabeta_action(state, 5, self.score)
        # self._iterative_alphabeta_action(state, 10, heuristic=self.score)

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

    def _iterative_alphabeta_action(self,state, depth_limit, heuristic):
        # Turns out "iterative deepening" is just a for loop...
        best_move = None
        for depth in range(1, depth_limit+1):
            best_move = self.alpha_beta_search(state, depth, heuristic)
            self.queue.put(best_move)

    def _minimax_alphabeta_action(self, state, depth, heuristic):
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
        self.queue.put(self.alpha_beta_search(state, depth, heuristic))

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
