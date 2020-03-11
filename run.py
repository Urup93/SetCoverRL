import torch
import numpy as np
from models import *
from Agent import *
from set_cover_env import *
from set_cover_env import _generate_instance
import os
from datetime import datetime
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import RandomTing.solver as solver


def draw_params(SL, SH, UL, UH):
    s = random.randint(SL, SH)
    u = random.randint(UL, UH)
    p = random.random()*0.2
    return s, u, p

SL, SH = 50, 50
UL, UH = 20, 20
s, u, p = draw_params(SL, SH, UL, UH)
env = SetCoverEnv(s, u, p)
model = SubsetRanking(input_uni_feat=16, input_sub_feat=32, output_uni_feat=16, output_sub_feat=32, n_hid=64)

agent = DDQN_Agent(env, model, 5, 40)
#agent = Q_Agent(env, model)

#os.system('wandb login d9cec596df561142f91b26f14ef3c278b84eae99')
#wandb.init(project='speciale')
#wandb.watch(model)


model_path = os.path.dirname(os.path.realpath(__file__))+'\model\model1.pt'
if os.path.isfile(model_path):
    model.load_model(model_path)

info = pd.DataFrame(columns=['Subsets', 'Elements', 'Edge prob', 'Seconds used', 'Loss', 'Solution'])


iterations = 1000
for i in range(iterations):
    print('Training on random graph: ', i+1, ' out of ', iterations)
    s, u, p = draw_params(SL, SH, UL, UH)
    env.reset(s, u, p)
    #print('Current graph has: ', s, ' subsets and ', u, ' elements')
    time = datetime.now()
    test_loss = agent.train()
    time = datetime.now() - time
    #wandb.log({'Test loss' : test_loss})

    info.loc[i] = [s, u, round(p, 2), time.seconds, round(test_loss, 2), sum(env.get_solution())]

    if (i+1)%25 == 0 or i == iterations-1:
        model.save_model(model_path)
        print('model saved on harddisk')
        #wandb.save(os.path.join(wandb.run.dir, 'checkpoint*'))
        #print('model saved on cloud')

        s, u, p = draw_params(SL, SH, UL, UH)
        inst = _generate_instance(u,s, p)
        env.set_instance(inst)

        agent.train()
        agent_sol = sum(env.get_solution())

        greedy = solver.greedySolver()
        greedy_sol, c = greedy.solve(inst, np.ones(s))

        plt.plot([i/25], [agent_sol-len(greedy_sol)], 'ro')


plt.show()
info.to_csv(os.path.dirname(os.path.realpath(__file__)) + '/model/training_info.csv', index=False, sep=',', header=True)
print('Dataframe saved')




