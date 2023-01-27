import numpy as np
import pandas as pd
import plotly.express as px
from copy import copy


def fitness(position:np.ndarray[float, float]):
    x, y = position
    return (1.5 - x - x*y)**2 + (2.25 - x + (x*y)**2)**2 + (2.625 - x + (x*y)**3)**2
    # return (1.5 - x - x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

def f(x:float, y:float):
    return (1.5 - x - x*y)**2 + (2.25 - x + (x*y)**2)**2 + (2.625 - x + (x*y)**3)**2
    # return (1.5 - x - x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

C1_PARAMETER = .9
C2_PARAMETER = 1
ITERATIONS = 50
POINTS_LIN_COUNT = 20
PLAIN_MAX_COORD = 4.5
MAX_VELOCITY = 1
GLOBAL_BEST = {"position": np.array((0,0)), "value":fitness((0,0))}


class Particle:
    def __init__(self, position:np.ndarray):        
        self.position = position
        self.personal_best = {"position": position, "value":fitness(position)}
        self.velocity = np.random.uniform(low=-MAX_VELOCITY, high=MAX_VELOCITY, size=2)        
    
    def update_position(self, change_rate:float):
        global GLOBAL_BEST
        self.velocity += ((self.personal_best["position"] - self.position) + (GLOBAL_BEST["position"] - self.position))
        self.velocity = np.clip(self.velocity, -MAX_VELOCITY, MAX_VELOCITY)
        
        self.position += np.random.random(size=2) * self.velocity * change_rate * np.array((C1_PARAMETER, C2_PARAMETER))
        self.position = np.clip(self.position, -PLAIN_MAX_COORD, PLAIN_MAX_COORD)
        
        func_value = fitness(self.position)
        if func_value < self.personal_best["value"]:
            self.personal_best = {"position": self.position, "value":func_value}
            if func_value < GLOBAL_BEST["value"]:
                GLOBAL_BEST = {"position": self.position, "value":func_value}
                # print(f"New global minimum: --==## {GLOBAL_BEST['value']} | {GLOBAL_BEST['position']} ##==--")
    
    def __str__(self) -> str:
        return f"Position: {self.position} | Personal best: {self.personal_best} | Velocity: {self.velocity}"


def funkcja_1(swarm:list[Particle], df:pd.DataFrame):
    current_state_df = pd.DataFrame((particle.position for particle in swarm), columns=("x_axis", "y_axis"))
    current_state_df["velocity"] = [copy(particle.velocity) for particle in swarm]
    current_state_df["iter"] = 0
    df = pd.concat([df, current_state_df], ignore_index=True)
    
    for iter in range(ITERATIONS):
        velocity_change_rate = 1 - (iter/ITERATIONS)
        for particle in swarm:
            particle.update_position(velocity_change_rate)
        
        current_state_df = pd.DataFrame((particle.position for particle in swarm), columns=("x_axis", "y_axis"))
        current_state_df["velocity"] = [particle.velocity for particle in swarm]
        current_state_df["iter"] = iter+1
        
        df = pd.concat([df, current_state_df], ignore_index=True)
    
    return df


if __name__ == "__main__":
    # np.random.seed(6)
    
    linspace = np.linspace(-PLAIN_MAX_COORD, PLAIN_MAX_COORD, POINTS_LIN_COUNT)
    swarm = [Particle(np.array((x, y))) for x in linspace for y in linspace]
    
    df = pd.DataFrame()
    df = funkcja_1(swarm, df)

    print(GLOBAL_BEST)
    
    fig = px.scatter(df, x="x_axis", y="y_axis", animation_frame="iter")
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 250
    fig.show()



"""# Background coloring

bars_linspace = np.linspace(-PLAIN_MAX_COORD, PLAIN_MAX_COORD, 1000)
points = [(x, y) for x in bars_linspace for y in bars_linspace]
bars_x = np.array([point[0] for point in points])
bars_y = np.array([point[1] for point in points])
bars_z = f(bars_x, bars_y)

bars_x, bars_y = np.array(np.meshgrid(bars_linspace, bars_linspace))
bars_z = f(bars_x, bars_y)
# print(bars_z)
# print(bars_z.shape())

x_min = bars_x.ravel()[bars_z.argmin()]
y_min = bars_y.ravel()[bars_z.argmin()]

plt.figure(figsize=(9,7))
plt.imshow(bars_z, extent=[-PLAIN_MAX_COORD, PLAIN_MAX_COORD]*2, alpha=0.5)  # colorful background
plt.plot([x_min], [y_min], marker="x", markersize=5, color="white")
contours = plt.contour(bars_x, bars_y, bars_z, 20, colors="black", alpha=0.4)
plt.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
plt.show()
"""
