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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.
    """

    def getAction(self, gameState):
        legalMoves = gameState.getLegalActions()
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        def minimax(agent_index, depth, game_state):
            if agent_index >= game_state.getNumAgents():
                agent_index = 0
                depth += 1
            if depth == self.depth or game_state.isWin() or game_state.isLose():
                return self.evaluationFunction(game_state), None

            legal_actions = game_state.getLegalActions(agent_index)
            if not legal_actions:
                return self.evaluationFunction(game_state), None

            if agent_index == 0:
                best_score, best_action = float('-inf'), None
                for action in legal_actions:
                    successor = game_state.generateSuccessor(agent_index, action)
                    score, _ = minimax(agent_index + 1, depth, successor)
                    if score > best_score:
                        best_score, best_action = score, action
                return best_score, best_action
            else:
                best_score, best_action = float('inf'), None
                for action in legal_actions:
                    successor = game_state.generateSuccessor(agent_index, action)
                    score, _ = minimax(agent_index + 1, depth, successor)
                    if score < best_score:
                        best_score, best_action = score, action
                return best_score, best_action

        _, action = minimax(0, 0, gameState)
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        """
        Returns the action using Alpha-Beta Pruning based on the pseudocode.
        """

        def max_value(state, alpha, beta, depth):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            v = float('-inf')
            for action in state.getLegalActions(0):
                v = max(v, min_value(state.generateSuccessor(0, action), alpha, beta, depth, 1))
                if v > beta:
                    return v
                alpha = max(alpha, v)

            return v

        def min_value(state, alpha, beta, depth, agentIndex):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            v = float('inf')
            next_agent = (agentIndex + 1) % state.getNumAgents()
            is_last_agent = next_agent == 0

            for action in state.getLegalActions(agentIndex):
                if is_last_agent:
                    v = min(v, max_value(state.generateSuccessor(agentIndex, action), alpha, beta, depth + 1))
                else:
                    v = min(v, min_value(state.generateSuccessor(agentIndex, action), alpha, beta, depth, next_agent))

                if v < alpha:
                    return v
                beta = min(beta, v)

            return v

        # Start alpha-beta pruning from the root (Pacman is MAX)
        alpha = float('-inf')
        beta = float('inf')
        best_action = None
        v = float('-inf')

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            successor_value = min_value(successor, alpha, beta, 0, 1)

            if successor_value > v:
                v = successor_value
                best_action = action

            alpha = max(alpha, v)

        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        def expectimax(agent_index, depth, game_state):
            if agent_index >= game_state.getNumAgents():
                agent_index = 0
                depth += 1
            if depth == self.depth or game_state.isWin() or game_state.isLose():
                return self.evaluationFunction(game_state), None

            legal_actions = game_state.getLegalActions(agent_index)
            if not legal_actions:
                return self.evaluationFunction(game_state), None

            if agent_index == 0:
                best_score, best_action = float('-inf'), None
                for action in legal_actions:
                    successor = game_state.generateSuccessor(agent_index, action)
                    score, _ = expectimax(agent_index + 1, depth, successor)
                    if score > best_score:
                        best_score, best_action = score, action
                return best_score, best_action
            else:
                expected_score = 0
                for action in legal_actions:
                    successor = game_state.generateSuccessor(agent_index, action)
                    score, _ = expectimax(agent_index + 1, depth, successor)
                    expected_score += score / len(legal_actions)
                return expected_score, None

        _, action = expectimax(0, 0, gameState)
        return action

def betterEvaluationFunction(currentGameState):
    return currentGameState.getScore()

better = betterEvaluationFunction
