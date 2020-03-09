import torch
import numpy as np
from models import *
from Agent import *
from set_cover_env import *
import os
from datetime import datetime
import pandas as pd
import wandb

def draw_params(SL, SH, UL, UH):
    s = random.randint(SL, SH)
    u = random.randint(UL, UH)
    p = random.random()
    return s, u, p

SL, SH = 100, 500
UL, UH = 100, 500
s, u, p = draw_params(SL, SH, UL, UH)
env = SetCoverEnv(s, u, p)
model = SubsetRanking(n_uni_feat=1, n_sub_feat=3, n_hid=64)
agent = DDQN_Agent(env, model, 5, 40)


os.system('wandb login d9cec596df561142f91b26f14ef3c278b84eae99')
wandb.init(project='speciale')
wandb.watch(model)


dir_path = os.path.dirname(os.path.realpath(__file__))
if os.path.isfile(dir_path):
    model.load_model(dir_path)


info = pd.DataFrame(columns=['Subsets', 'Elements', 'Edge prob', 'Seconds used', 'Loss', 'Solution'])


iterations = 1000
for i in range(iterations):
    print('Training on random graph: ', i+1, ' out of ', iterations)
    time = datetime.now()
    test_loss = agent.train()
    time = datetime.now() - time

    wandb.log({'Test loss' : test_loss})

    info.loc[i] = [s, u, round(p, 2), time.seconds, round(test_loss, 2), sum(env.get_solution())]

    s, u, p = draw_params(SL, SH, UL, UH)
    env.reset(s, u, p)

    if (i+1)%25 == 0 or i == iterations-1:
        model.save_model(dir_path+'/model/model1.pt')
        print('model saved on harddisk')
        wandb.save(os.path.join(wandb.run.dir, 'checkpoint*'))
        print('model saved on cloud')

info.to_csv(dir_path+'/model/training_info.csv', index=False, sep=',', header=True)
print('Dataframe saved')