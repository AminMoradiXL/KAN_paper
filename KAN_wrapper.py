from kan import *
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error as rmse
import warnings
warnings.filterwarnings("ignore", message="std()*")
import os 
from matplotlib import pyplot as plt
from datetime import datetime
from aux_func import *
from sklearn.model_selection import train_test_split






name = 'Ikeda'  # Ikeda, Logistic, Circle, Henon, food_chain_system

path = 'data/'+ name + '.pkl'
data_path = f"data"

now = datetime.now()
res_path = f"results/{name}/{name}_{now.strftime('%m-%d-%H%M%S')}"

if not os.path.exists(data_path):
    os.makedirs(data_path)

if not os.path.exists(res_path):
    os.makedirs(res_path)

    
if not os.path.exists(path):
    print('Generating data. This might take a bit!')
    try: 
        data_gen(name)
        print('Data generated succesfuly!')
        df = pd.read_pickle(path)
        print('Data Loaded succesfuly!')
        print('Plotting data!')
        plot(name, df)
    except Exception as e:
        print("Model doesn't exist!" )
        print("Choose one of the implemented functions: 'Logistic', 'Henon', 'Ikeda', 'Circle', 'food_chain_system'")
else:
    print('Loading previously generated data')
    df = pd.read_pickle(path)
    print('Data Loaded succesfuly!')
    print('Plotting data!')
    plot(name, df)


    
    
#################  Parameter Ikeda

# data Ikeda1
transient_perc = 0.01
test_perc = 0.2
n_ms = 10000
t_plot_train = 150
t_plot_test = 75


# KAN
hidden_layer = [4]
grid_size = 10
k_size = 3
random_seed = 0
beta = 100
iterations = 50
batch_size = -1
lamb = 0
lamb_entropy = 10
# learning_rate = 0.1


# #################  Parameter food_chain

# # data food_chain
# transient_perc = 0.05
# test_perc = 0.25
# n_ms = 5000
# t_plot_train = 1000
# t_plot_test = 700


# # KAN
# hidden_layer = []
# grid_size = 3
# k_size = 3
# random_seed = 0
# beta = 100
# iterations = 100
# batch_size = -1
# lamb = 0
# lamb_entropy = 10
# learning_rate = 0.5



# save parameters

with open(f'{res_path}/parameters.txt', 'w') as f:
     f.write(f"transient_perc: {transient_perc}\n")
     f.write(f"test_perc: {test_perc}\n")
     f.write(f"n_ms: {n_ms}\n")
     f.write(f"grid_size: {grid_size}\n")
     f.write(f"k_size: {k_size}\n")
     f.write(f"random_seed: {random_seed}\n")
     f.write(f"hidden_layer: {hidden_layer}\n")
     f.write(f"beta: {beta}\n")
     f.write(f"iterations: {iterations}\n")
     f.write(f"lamb: {lamb}\n")
     f.write(f"lamb_entropy: {lamb_entropy}\n")
     f.write(f"learning_rate: {learning_rate}\n")


dataset, df_seen, df_unseen, n_ms, in_vars, out_vars, dim = dataset_gen(df, transient_perc, test_perc, n_ms)



in_out = [int(dim), int(dim)]
kan_layers = in_out[:1] + hidden_layer + in_out[1:]
model = KAN(width=kan_layers, grid=grid_size, k=k_size, seed=random_seed)



## Train
model(dataset['train_input'])



plt.figure()
model.plot(beta=beta)
plt.savefig(f'{res_path}/initial_kan_model.svg')



results = model.train(dataset, opt="LBFGS", steps=iterations, batch = batch_size,
                      lamb=lamb, lamb_entropy=lamb_entropy, beta=beta, lr = learning_rate,
                      save_fig=False);

model.save_ckpt(f'{name}', folder = f'{res_path}/')  # Checkpoint save

plt.figure()
model.plot(beta=beta, scale=5)
plt.savefig(f'{res_path}/kan_model.svg')




## Training error
y_test_arr = dataset['test_label'].detach().numpy()
y_pred_arr = model(dataset['test_input']).detach().numpy()
error_seen = rmse(y_test_arr, y_pred_arr) 
print(f'1-step Prediction error is = {error_seen}')

# train_error
plt.figure()
plt.plot(results['train_loss'], label = 'train_loss')
plt.plot(results['test_loss'], label = 'val_loss')
plt.legend()
plt.tight_layout()
plt.savefig(f'{res_path}/train_error.svg', dpi =600)
plt.show()


plot_test(y_test_arr, y_pred_arr, dim, path=res_path, title='One Step Prediction', t = t_plot_train)



#  ### Multi-step

X_unseen = df_unseen[in_vars[0]].to_numpy()
y_unseen = df_unseen[out_vars[0]].to_numpy()
y_pred_unseen = np.zeros((n_ms,int(dim)))
# y_pred_unseen[0,:] = model(torch.tensor(X_unseen[0,:].reshape(1, -1))).detach().numpy()
y_pred_unseen[0,:] = model(torch.tensor(X_unseen[0,:].reshape(1, -1))).detach().numpy()

for i in range(n_ms-1):
    model.load_ckpt(f'{name}', folder = f'{res_path}/') # Checkpoint load
    pred = model(torch.tensor(y_pred_unseen[i,:].reshape(1, -1))).detach().numpy()
    # pred[0][1] = y_unseen[i+1][1]
    y_pred_unseen[i+1, :] = pred
     
y_unseen_trun = y_unseen[:n_ms,:]

plot_test(y_unseen_trun, y_pred_unseen, dim, path=res_path, title='Multi Step Prediction', t = t_plot_test)



save_result_path=res_path+'/save_results'
os.makedirs(save_result_path)
#Error

np.savetxt(f'{save_result_path}/train_loss.csv', np.array(results['train_loss']), delimiter=',')
np.savetxt(f'{save_result_path}/test_loss.csv', np.array(results['test_loss']), delimiter=',')
# 1-step
np.savetxt(f'{save_result_path}/y_train_1s.csv', y_test_arr, delimiter=',')
np.savetxt(f'{save_result_path}/y_pred_1s.csv', y_pred_arr, delimiter=',')
# multi-step
np.savetxt(f'{save_result_path}/y_train_ms.csv', y_unseen_trun, delimiter=',')
np.savetxt(f'{save_result_path}/y_pred_ms.csv', y_pred_unseen, delimiter=',')



