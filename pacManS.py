# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import math
import statistics


from pacman import GameState
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def get_value(self, game_state, current_player, depth):
        if game_state.is_terminal():
            return better_evaluation_function(game_state)

        if current_player % 3 == 0:
            if game_state.is_terminal():
                return better_evaluation_function(game_state)
            available_actions = game_state.get_valid_actions(player_index=0)
            highest_value = float('-inf')
            random.shuffle(available_actions)  # Randomize the order of valid actions
            for action in available_actions:
                highest_value = max(
                    self.get_value(game_state.generate_successor(player_index=0, action=action), current_player + 1,
                                   depth), highest_value)
            return highest_value
        else:
            available_actions = game_state.get_valid_actions(player_index=current_player % 3)
            lowest_value = float('+inf')
            random.shuffle(available_actions)  # Randomize the order of valid actions
            for action in available_actions:
                if current_player % 3 == 2:
                    lowest_value = min(
                        self.get_value(game_state.generate_successor(player_index=current_player % 3, action=action),
                                       current_player + 1, depth + 1), lowest_value)
                else:
                    lowest_value = min(
                        self.get_value(game_state.generate_successor(player_index=current_player % 3, action=action),
                                       current_player + 1, depth), lowest_value)
            return lowest_value

    def getAction(self, state):
        valid_actions = state.getValidActions(current_player=0)
        highest_value = float('-inf')
        best_action = valid_actions[0]

        random.shuffle(valid_actions)  # Randomize the order of valid actions
        for action in valid_actions:
            next_states = state.generateSuccessor(current_player=0, action=action)
            current_value = self.value(next_states, next_player=current_player + 1, depth=0)

            # Introduce some randomness in the decision-making process
            if current_value > highest_value or (current_value == highest_value and random.random() < 0.5):
                best_action = action
                highest_value = current_value

        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState:GameState):
    def minDistanceBfs(currentGameState:GameState):
        walls = currentGameState.getWalls()
        height = 0
        for i in walls:
            height+=1
        width = 0
        for i in walls[0]:
            width += 1

        start_position = currentGameState.getPacmanPosition()
        visited = set()
        queue = util.Queue()
        queue.push([start_position,0])
        while not queue.isEmpty():
            sposition = queue.pop()
            x , y = sposition[0]
            if currentGameState.hasFood(x , y):
                return sposition[1]
            if sposition[0] in visited:
                continue
            visited.add(sposition[0])

            x , y = sposition[0]

            if not walls[x-1][y] and x > 0:
                queue.push([(x-1,y),sposition[1]+1])
            if not walls[x+1][y] and x < height:
                queue.push([(x+1,y),sposition[1]+1])
            if not walls[x][y-1] and y > 0:
                queue.push([(x,y-1),sposition[1]+1])
            if not walls[x][y+1] and y < width:
                queue.push([(x,y+1),sposition[1]+1])
        return float('inf')


    if currentGameState.isWin():
        return 50000 + currentGameState.getScore()
    if currentGameState.isLose():
        return -500000


    numFood = currentGameState.getNumFood()

    return currentGameState.getScore()/5 - 50 * numFood + 9/minDistanceBfs(currentGameState)
# Abbreviation
better = betterEvaluationFunction