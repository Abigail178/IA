#==================================================
# Agente Viborita Inteligente
#==================================================
import torch
import random
import numpy as np
from collections import deque
from juego import ViboritaInteligente, Direccion, Punto
from modelo import Linear_QNet, QTrainer
from grafica import graficar

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
JUEGOS_AL_AZAR = 80

#==========================
# Clase Agente
#==========================
class Agent:
    
    #===============================
    # Constructor:
    #   model - red neuronal
    #   trainer - optimizador
    #===============================
    def __init__(self):
        self.n_games = 0
        self.epsilon = JUEGOS_AL_AZAR # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
    #===========================
    # Estado del agente
    #===========================
    def obtener_estado(self, juego):
        
        head = juego.snake[0]
        
        point_l = Punto(head.x - 20, head.y)
        point_r = Punto(head.x + 20, head.y)
        point_u = Punto(head.x, head.y - 20)
        point_d = Punto(head.x, head.y + 20)
        
        dir_l = juego.direction == Direccion.LEFT
        dir_r = juego.direction == Direccion.RIGHT
        dir_u = juego.direction == Direccion.UP
        dir_d = juego.direction == Direccion.DOWN

        state = [
            #========================
            # Danger straight
            #========================
            (dir_r and juego.is_collision(point_r)) or 
            (dir_l and juego.is_collision(point_l)) or 
            (dir_u and juego.is_collision(point_u)) or 
            (dir_d and juego.is_collision(point_d)),

            #=====================
            # Danger right
            #=====================
            (dir_u and juego.is_collision(point_r)) or 
            (dir_d and juego.is_collision(point_l)) or 
            (dir_l and juego.is_collision(point_u)) or 
            (dir_r and juego.is_collision(point_d)),

            #=================
            # Danger left
            #=================
            (dir_d and juego.is_collision(point_r)) or 
            (dir_u and juego.is_collision(point_l)) or 
            (dir_r and juego.is_collision(point_u)) or 
            (dir_l and juego.is_collision(point_d)),
            
            #===================
            # Move direction
            #===================
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            #===================
            # Food location 
            #===================
            juego.food.x < juego.head.x,  # food left
            juego.food.x > juego.head.x,  # food right
            juego.food.y < juego.head.y,  # food up
            juego.food.y > juego.head.y  # food down
            ]

        return np.array(state, dtype=int)

    #===========================
    # Añadir en memoria
    #===========================
    def recordar(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    #=================================
    # Entrenar memoria de largo plazo
    #=================================
    def entrenar_memoria_larga(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    #=======================================
    #Entrenar memoria de corto plazo
    #========================================
    def entrenar_memoria_corta(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


    #=================
    # Decidir accion
    #=================
    def obtener_accion(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = JUEGOS_AL_AZAR - self.n_games
        final_move = [0,0,0]
        
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


#===================================
# FUNCION PRINCIPAL: ENTRENAMIENTO
#===================================
def entrenar():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    juego = ViboritaInteligente()
    
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.entrenar_memoria_corta(state_old, final_move, reward, state_new, done)

        # remember
        agent.recordar(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            juego.reset()
            agent.n_games += 1
            agent.entrenar_memoria_larga()

            if score > record:
                record = score
                agent.model.save()

            print('Juego', agent.n_games, 'Puntos', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            graficar(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    entrenar()