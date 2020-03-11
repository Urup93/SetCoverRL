import os
from Agent import *
from models import *
from set_cover_env import *
import RandomTing.solver as solver
from set_cover_env import _generate_instance

inst = _generate_instance(20, 50, 0.2)

s, u, p = 20, 50, random.random()*0.2
env = SetCoverEnv(s, u, p)
model = SubsetRanking(input_uni_feat=16, input_sub_feat=32, output_uni_feat=16, output_sub_feat=32, n_hid=64)
model_path = os.path.dirname(os.path.realpath(__file__)) + '\model\model1.pt'
assert os.path.isfile(model_path), 'no model loaded'
model.load_model(model_path)
model.eval()
env.set_instance(inst)
agent = DDQN_Agent(env, model, 5, 40)


print('evaling model')
sol = agent.eval()
print('sol: ', sum(env.get_solution()))

print('computing greedy solution')
greedy = solver.greedySolver()
greedy_sol, c = greedy.solve(inst, np.ones(50))

print('Greedy solution: ', len(greedy_sol))


