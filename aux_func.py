import torch
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error as rmse
import warnings
warnings.filterwarnings("ignore", message="std()*")
import os 
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split



def df(name, data):
    dim = np.shape(data)[1]
    
    if dim == 1:
        # df = pd.DataFrame(data[:-1,:], columns=['x_n'])
        df = pd.DataFrame(data[:-1], columns=['x_n'])
        df['x_n+1'] = data[1:]
    elif dim == 2: 
        df = pd.DataFrame(data[:-1,:], columns=['x_n', 'y_n'])
        df['x_n+1'] = data[1:,0]
        df['y_n+1'] = data[1:,1]
    elif dim == 3: 
        df = pd.DataFrame(data[:-1,:], columns=['x_n', 'y_n', 'z_n'])
        df['x_n+1'] = data[1:,0]
        df['y_n+1'] = data[1:,1]
        df['z_n+1'] = data[1:,2]
        
    df.to_pickle(f'data/{name}.pkl')

    
def data_gen(name):
    class_object = globals()[name]
    obj = class_object()
    df(name, obj)
    
 
def dataset_gen(df, transient_perc, test_perc, n_ms):
    
    transient_to_drop = int(len(df) * transient_perc)
    df = df.iloc[-int(len(df)*transient_perc)-1:,:]
    df_seen = df.iloc[:int((1-test_perc)*len(df)),:]
    df_unseen = df.iloc[int((1-test_perc)*len(df)):,:]

    if len(df_unseen)<n_ms:
        n_ms = len(df_unseen)
        
    dim = np.shape(df)[1]/2
    if dim == 1:
        X = df_seen[['x_n']].to_numpy()
        y = df_seen[['x_n+1']].to_numpy() 
        in_vars=['x_n'],
        out_vars=['x_n+1'],
    elif dim == 2:
        X = df_seen[['x_n','y_n']].to_numpy()
        y = df_seen[['x_n+1','y_n+1']].to_numpy()
        in_vars=['x_n', 'y_n'],
        out_vars=['x_n+1', 'y_n+1'],
    elif dim == 3:
        X = df_seen[['x_n','y_n', 'z_n']].to_numpy()
        y = df_seen[['x_n+1','y_n+1', 'z_n+1']].to_numpy() 
        in_vars=['x_n', 'y_n', 'z_n'],
        out_vars=['x_n+1', 'y_n+1', 'z_n+1'],
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size= 0.2, random_state= 42, shuffle = False)

    dataset = {
        "train_input": torch.tensor(X_train),
        "train_label": torch.tensor(y_train),
        "test_input": torch.tensor(X_val), 
        "test_label": torch.tensor(y_val)
    }

    return dataset, df_seen, df_unseen, n_ms, in_vars, out_vars, dim
    
def Ikeda(N_sims= 1, T= 1e6, dt=1, a=0.4, u=0.9, x0=0.1, y0=0.1):
    N_t = int(T // dt)
    sims = np.zeros((N_sims, N_t, 2))
    sims[:, 0] = np.array([x0, y0]).T
    for i in range(1, N_t):
        theta = 0.4 - (6 / (1 + sims[:, i-1, 0]**2 + sims[:, i-1, 1]**2))
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        sims[:, i] = np.array([1 + u * (sims[:, i-1, 0] * cos_theta - sims[:, i-1, 1] * sin_theta),
                               u * (sims[:, i-1, 0] * sin_theta + sims[:, i-1, 1] * cos_theta)]).T
    return sims.astype(np.float32)[0]

def Logistic(N_sims= 1, T= 1e6, dt=1, a=4, x0=0.1):
    N_t = int(T // dt)
    sims = np.zeros((N_sims, N_t, 1))
    sims[:, 0] = np.array([x0]).T
    for i in range(1, N_t):
        sims[:, i] = np.array(a * (sims[:, i-1])* (1-sims[:, i-1]))                
    return sims.astype(np.float32)[0]

def Circle(N_sims= 1, T= 1e6, dt=1, x0=0.1, K=1, omega=0.3):
    num_steps = int(T // dt)
    x_values = np.zeros(num_steps)
    
    x_values[0] = x0
    
    for i in range(1, num_steps):
        x_values[i] = (x_values[i-1] + omega - (K/(2*np.pi))*np.sin(2*np.pi*x_values[i-1])) % 1
    
    return np.vstack((x_values))


def Henon(N_sims= 1, T= 1e6, dt=1, x0=0.1, y0=0.1, a=1.4, b=0.3):
    num_steps = int(T // dt)
    x_values = np.zeros(num_steps)
    y_values = np.zeros(num_steps)
    
    x_values[0] = x0
    y_values[0] = y0
    
    for i in range(1, num_steps):
        x_values[i] = 1 - a * x_values[i-1]**2 + y_values[i-1]
        y_values[i] = b * x_values[i-1]
    
    return np.vstack((x_values, y_values)).T


def food_chain_system(x0=0.9796, y0=0.2687, z0=0.7297, num_steps=int(1e5), dt=5e-1):
    
    xc=0.4
    yc=2.009
    xp=0.08
    yp=2.876
    r0=0.16129
    c0=0.5
    def food_chain_equations(x, y, z):
        dx_dt = x * (1 - x/k) - xc * yc * y * x / (x + r0)
        dy_dt = xc * y * (yc * x / (x + r0) - 1) - xp * yp * z * y / (y + c0)
        dz_dt = xp * z * (yp * y / (y + c0) - 1)
        return dx_dt, dy_dt, dz_dt
    
    k = 0.98  # Carrying capacity for the prey
    x_values = np.zeros(num_steps)
    y_values = np.zeros(num_steps)
    z_values = np.zeros(num_steps)
    
    x_values[0] = x0
    y_values[0] = y0
    z_values[0] = z0
    
    for i in range(1, num_steps):
        k1_x, k1_y, k1_z = food_chain_equations(x_values[i-1], y_values[i-1], z_values[i-1])
        k2_x, k2_y, k2_z = food_chain_equations(x_values[i-1] + 0.5 * dt * k1_x,
                                                y_values[i-1] + 0.5 * dt * k1_y,
                                                z_values[i-1] + 0.5 * dt * k1_z)
        k3_x, k3_y, k3_z = food_chain_equations(x_values[i-1] + 0.5 * dt * k2_x,
                                                y_values[i-1] + 0.5 * dt * k2_y,
                                                z_values[i-1] + 0.5 * dt * k2_z)
        k4_x, k4_y, k4_z = food_chain_equations(x_values[i-1] + dt * k3_x,
                                                y_values[i-1] + dt * k3_y,
                                                z_values[i-1] + dt * k3_z)
        
        x_values[i] = x_values[i-1] + (dt / 6) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        y_values[i] = y_values[i-1] + (dt / 6) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
        z_values[i] = z_values[i-1] + (dt / 6) * (k1_z + 2*k2_z + 2*k3_z + k4_z)
    
    return np.vstack((x_values, y_values, z_values)).T
    
def plot(name, df):
    
    fig = plt.figure(figsize=(8, 6))
    dim = int(np.shape(df)[1]/2)
    
    trajectories = df.to_numpy().astype(np.float64)[:,0:dim]
        
    if dim == 1:
        plt.scatter(trajectories[0:-1], trajectories[1:], s=1)
        plt.title(f"{name} Trajectories (Scatter Plot)")
        plt.xlabel("X_n")
        plt.ylabel("X_n+1")
        plt.show()
    elif dim == 2:
        plt.scatter(trajectories[:, 0], trajectories[:, 1], s=1)
        plt.title(f"{name} Trajectories (Scatter Plot)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
    elif dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(trajectories[:, 0], trajectories[:, 1], trajectories[:, 2], s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"{name} Trajectories (Scatter Plot)")
        plt.show()
        


def plot_test(real_arr, pred_arr, dim, path, title, t = 100): # For plotting results
    
    if t>len(real_arr):
        t = len(real_arr) -1
    
    if dim == 1:
        
        fig, axs = plt.subplots(2, figsize=(8, 10))
        
        # Subplot 1: Attractor Learning
        axs[0].scatter(real_arr[0:-1], real_arr[1:], s=1, label='real')
        axs[0].scatter(pred_arr[0:-1], pred_arr[1:], s=1, label='pred')
        axs[0].set_title('Attractor')
        axs[0].legend()
        
        # Subplot 2: Time Series Learning
        axs[1].plot(real_arr[:t], label='real')
        axs[1].plot(pred_arr[:t], label='pred')
        axs[1].set_title('Time Series')
        axs[1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{path}/{title}.svg', dpi =600)
        plt.show()
         
    elif dim == 2: 
        
        fig, axs = plt.subplots(3, figsize=(8, 15))
        
        axs[0].scatter(real_arr[:, 0], real_arr[:, 1], s=1, label='real')
        axs[0].scatter(pred_arr[:, 0], pred_arr[:, 1], s=1, label='pred')
        axs[0].set_title('Attractor')
        axs[0].legend()
        
        axs[1].plot(real_arr[:t, 0], label='real')
        axs[1].plot(pred_arr[:t, 0], label='pred')
        axs[1].set_title('Time Series - Dimension 1')
        axs[1].legend()
        
        
        axs[2].plot(real_arr[:t, 1], label='real')
        axs[2].plot(pred_arr[:t, 1], label='pred')
        axs[2].set_title('Time Series - Dimension 2')
        axs[2].legend()
        
        plt.tight_layout()
        plt.savefig(f'{path}/{title}.svg', dpi =600)
        plt.show()

        
    elif dim == 3: 
        
        fig = plt.figure(figsize=(12, 10))
        
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.scatter(real_arr[:, 0], real_arr[:, 1], real_arr[:, 2], s=1, label='real')
        ax1.scatter(pred_arr[:, 0], pred_arr[:, 1], pred_arr[:, 2], s=1, label='pred')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title("Attractor")
        ax1.legend()
        
        ax2 = fig.add_subplot(222)
        ax2.plot(real_arr[:t, 0], label='real')
        ax2.plot(pred_arr[:t, 0], label='pred')
        ax2.set_title('Time Series - Dimension 1')
        ax2.legend()
        
        ax3 = fig.add_subplot(223)
        ax3.plot(real_arr[:t, 1], label='real')
        ax3.plot(pred_arr[:t, 1], label='pred')
        ax3.set_title('Time Series - Dimension 2')
        ax3.legend()
        
        ax4 = fig.add_subplot(224)
        ax4.plot(real_arr[:t, 2], label='real')
        ax4.plot(pred_arr[:t, 2], label='pred')
        ax4.set_title('Time Series - Dimension 3')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(f'{path}/{title}.svg', dpi =600)
        plt.show()
        