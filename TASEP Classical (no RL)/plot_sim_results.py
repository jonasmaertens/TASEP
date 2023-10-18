import matplotlib.pyplot as plt
import pandas as pd


dataPy = pd.read_csv('sim_results_py.csv')
data_cpp = pd.read_csv('sim_results_cpp.csv')
dt_py = dataPy['t']
C_py = dataPy['C']
C_times_dt_py = dataPy['C*t']
dt_cpp = data_cpp['t']
C_cpp = data_cpp['C']
C_times_dt_cpp = data_cpp['C*t']

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('C', color=color)
ax1.plot(dt_py, C_py, label='C_py')
ax1.plot(dt_py, C_cpp, label='C_cpp')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  

color = 'tab:blue'
ax2.set_ylabel('C*tau', color=color) 
ax2.plot(dt_py, C_times_dt_py, label='C_times_dt_py')
ax2.plot(dt_cpp, C_times_dt_cpp, label='C_times_dt_cpp')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout() 

ax1.legend()
ax2.legend()
#plt.xlim(1, 25)
plt.show()

