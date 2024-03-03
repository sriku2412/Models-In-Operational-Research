#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Question 1
print("Question 1")

import pandas as pd

df = pd.read_csv('price_response.csv')

from gurobipy import GRB, quicksum, Model
import gurobipy as gb
import numpy as np

#Question 1 Part A

a1, b1, c1 = df.at[0, 'Intercept'], df.at[0, 'Sensitivity'], df.at[0, 'Capacity']
a2, b2, c2 = df.at[1, 'Intercept'], df.at[1, 'Sensitivity'], df.at[1, 'Capacity']

model = gb.Model('TechEssentialPricing')

p1 = model.addVar(lb=0, name ='p1')
p2 = model.addVar(lb=0, name='p2')

model.setObjective(p1 * (a1 + b1 * p1) + p2 * (a2 + b2 * p2), GRB.MAXIMIZE)

model.addConstr(p2 >= p1, 'PriceOrdering')


model.addConstr(a1+b1*p1 >= 0,'DemandNonNegativity1')
model.addConstr(a2+b2*p2 >= 0,'DemandNonNegativity2')
model.addConstr(a1+b1*p1 <= c1,'capacityLimit1')
model.addConstr(a2+b2*p2 <= c2,'capacityLimit2')


model.optimize()

print('Part a: \n')
print(f'Optimal price for Basic version (p1): ${p1.X}')
print(f'Optimal price for Advanced version (p2): ${p2.X}')

    
#Question 2 Part B

#Initialize prices
p = np.array([0.0, 0.0])# starting with all prices set to zero
alpha =0.001 # Step size 
tolerance =1e-6
max_iterations = 2000 # Just as a safeguard, to avoid infinite loops 

def gradient(p):
    return np.array([a1 + 2*b1*p[0], a2 + 2*b2*p[1]]) # differential of the objective function w.r.t p
# a+b.p +p(b)

for iteration in range(max_iterations): 
    p += alpha * gradient(p)
    if np.allclose(np.abs(gradient(p) - p), 0, atol=tolerance):
        break

print('Part b: \n')
print(f'Optimal prices after gradient descent: p1 = ${p[0]}，p2 = ${p[1]}') 

#Part C:
model = gb.Model('Optimal Pricing')

prices = model.addVars(len(df), lb=0, name = 'Price')

quantities = model.addVars(len(df), lb=0, name = 'Quantity')

model.setObjective(quicksum(prices[i] * quantities[i] for i in range(len(df))),GRB.MAXIMIZE)

for i in range(len(df)):
    model.addConstr(quantities[i] == df.loc[i, 'Intercept'] + df.loc[i, 'Sensitivity'] * prices[i])
    

for i in range(len(df)):
    model.addConstr(quantities[i] <= df.loc[i, 'Capacity'])
    
    
model.optimize()

optimized_prices = model.getAttr('x', prices)
optimized_quantities = model.getAttr('x', quantities)

total_revenue = sum(optimized_prices[i] * optimized_quantities[i] for i in range(len(df)))

print("\n Part c: \n")
for i in range(len(df)):
    revenue = optimized_prices[i] * optimized_quantities[i]
    print(f"Product {i+1} Revenue: {revenue}")
print("Optimal Total Revenue: ", total_revenue)

#Part D:
model = Model('Optimal Pricing')

prices = model.addVars(len(df), lb=0, name='Price')
quantities = model.addVars(len(df), lb=0, name='Quantity')

model.setObjective(quicksum(prices[i] * quantities[i] for i in range(len(df))), GRB.MAXIMIZE)

# Quentity depends on intercept, sensitivity, and price
for i in range(len(df)):
    model.addConstr(quantities[i] == df.loc[i, 'Intercept'] + df.loc[i, 'Sensitivity'] * prices[i])

# Quantity <= Capacity
for i in range(len(df)):
    model.addConstr(quantities[i] <= df.loc[i, 'Capacity'])


num_lines = 3  # Number of product lines
versions_per_line = 3  # Number of versions per product line

# Check if the dataframe has the expected number of rows
assert len(df) == num_lines * versions_per_line, "DataFrame does not have the expected number of rows."

# Add constraints within each product line
for line in range(num_lines):
    base_index = line * versions_per_line
    model.addConstr(prices[base_index] <= prices[base_index + 1])  # Basic price <= Advanced price
    model.addConstr(prices[base_index + 1] <= prices[base_index + 2])  # Advanced price <= Premium price

# Add constraints across product lines for the same version
for version in range(versions_per_line):
    model.addConstr(prices[version] <= prices[version + versions_per_line])  # EvoTech <= InfiniteEdge for same version
    model.addConstr(prices[version + versions_per_line] <= prices[version + 2 * versions_per_line])  # InfiniteEdge <= FusionBook for same version

model.optimize()


optimized_prices = model.getAttr('x', prices)
optimized_quantities = model.getAttr('x', quantities)

total_revenue = sum(optimized_prices[i] * optimized_quantities[i] for i in range(len(df)))

print("\n Part d: \n")
for i in range(len(df)):
    revenue = optimized_prices[i] * optimized_quantities[i]
    print(f"Product {i+1} Revenue: {revenue}")
print("Optimal Total Revenue: ", total_revenue)


# Question 2
print("Question 2")
# -*- coding: utf-8 -*-

import gurobipy as gp

# loading the data
players = pd.read_csv('BasketballPlayers.csv',index_col=0)
position_mapping = {'G/F': ['G', 'F'], 'F/C': ['F', 'C'], 'F': ['F'], 'G': ['G']}

players['Position'] = players['Position'].apply(lambda x: position_mapping.get(x, x))
players['forward'] = players['Position'].apply(lambda x: 1 if 'F' in x else 0)
players['center'] = players['Position'].apply(lambda x: 1 if 'C' in x else 0)
players['guard'] = players['Position'].apply(lambda x: 1 if 'G' in x else 0)
players.drop(['Position'], axis=1, inplace=True)

# Values from the dataset
skills = players.columns.drop(['forward', 'center', 'guard']).tolist()


def model_run(invite,flag):
    # Initializing the model
    global model
    model = gp.Model("Basketball Players selection")
    model.update()
    if flag == 0:
        model.setParam('OutputFlag', 0)
  
    # Decision variables
    global x
    x = model.addVars(players.index, vtype=gp.GRB.BINARY, name="x", lb=0, ub=1)
    # Auxiliary variable for the total number of players

    # constraint
    # Total players constraint
    model.addConstr((gp.quicksum(x[i] for i in players.index) == invite), name="total_players")

    # position constraints
    model.addConstr(gp.quicksum(x[i] * players.loc[i, 'guard'] for i in players.index) >= 0.3*invite, name="position Guard")
    model.addConstr(gp.quicksum(x[i] * (players.loc[i, 'forward'] + players.loc[i, 'center']) for i in players.index) >= 0.4*invite, name="position Forward")

    # Skill rating constraints
    for skill in skills:
        model.addConstr(gp.quicksum(x[i] * players.loc[i, skill] for i in players.index) >= 2.05 * invite)


    # Mutual exclusion and inclusion constraints
    # --------------------------------------------
    # Create an auxiliary binary variable to indicate if the sum is greater than 0
    aux_var = model.addVar(vtype=gp.GRB.BINARY, name="aux_var")

    # Constraint to activate aux_var 
    model.addConstr(gp.quicksum(x[i] for i in range(20, 25)) - 0.0001 <= aux_var * gp.GRB.INFINITY)

    # Constraint to deactivate aux_var when the sum is not greater than 0
    model.addConstr(gp.quicksum(x[i] for i in range(20, 25)) >= 0.0001 - (1 - aux_var) * gp.GRB.INFINITY)

    # Use aux_var in the addGenConstrIndicator method
    model.addGenConstrIndicator(aux_var, 1, gp.quicksum(x[i] for i in range(72, 79)) <= 1)
    # --------------------------------------------
    # Create an auxiliary binary variable for the first condition
    aux_var1 = model.addVar(vtype=gp.GRB.BINARY, name="aux_var1")

    # Link aux_var1 to the condition gp.quicksum(x[i] for i in range(105, 115)) > 0
    model.addConstr(gp.quicksum(x[i] for i in range(105, 115)) - 0.0001 <= aux_var1 * gp.GRB.INFINITY)
    model.addConstr(gp.quicksum(x[i] for i in range(105, 115)) >= 0.0001 - (1 - aux_var1) * gp.GRB.INFINITY)

    # Create two more auxiliary binary variables for the then conditions
    aux_var2 = model.addVar(vtype=gp.GRB.BINARY, name="aux_var2")
    aux_var3 = model.addVar(vtype=gp.GRB.BINARY, name="aux_var3")

    # Link these variables to their respective conditions
    model.addConstr(gp.quicksum(x[i] for i in range(45, 50)) >= 0.0001 - (1 - aux_var2) * gp.GRB.INFINITY)
    model.addConstr(gp.quicksum(x[i] for i in range(45, 50)) - 0.0001 <= aux_var2 * gp.GRB.INFINITY)
    model.addConstr(gp.quicksum(x[i] for i in range(65, 70)) >= 0.0001 - (1 - aux_var3) * gp.GRB.INFINITY)
    model.addConstr(gp.quicksum(x[i] for i in range(65, 70)) - 0.0001 <= aux_var3 * gp.GRB.INFINITY)

    # Add indicator constraints to enforce the conditional logic
    model.addGenConstrIndicator(aux_var1, 1, aux_var2 == 1)
    model.addGenConstrIndicator(aux_var1, 1, aux_var3 == 1)


    # --------------------------------------------
    # At least one player from each decile
    for i in range(1, 151, 10): #step function
        model.addConstr(gp.quicksum(x[j] for j in players.index if j >= i and j < i+10) >= 1)
    # Objective function (maximizing total skill rating)
    model.setObjective(gp.quicksum(x[i] * players.loc[i, skills].sum() for i in players.index), gp.GRB.MAXIMIZE)

    # Solve the model
    model.optimize()   
        

# --------------------------------------------
# Understanding the problem
print("Q2 - Basketball Players")
print('-'*100)
print("(a) What types of decision variables are needed to solve the problem?")
print("Binary decision variable is used to indicate whether a player is invited or not to the training camp.")
print('-'*100)
print("(b) How many decision variables must be included in the model?")
print("There are 150 players of which we need to invite 21 players to the training camp.")
print("Therefore, we need 150 binary decision variables for each player.")
print('-'*100)
print("(c) What is the objective function? Defend why this is an appropriate choice.")
print("The objective function is to maximize the skill rate choose to prepare the best team for the Olympics.")
model_run(21,1)

print('-'*100)
print("(d) Write down the following constraint in mathematical notation but only for player 72: If any player from 20-24 (inclusive) is invited, all players from 72-78 (inclusive) cannot be.")
print("If sum of x[20:25]>0 trigger-> sum of x[72:79]<1")
print("Adding a trigger variable to the model that makes this logic possible.")
print("We use grb.INFINITY to include all values of sum & use it like a logical function.")
print("The subtraction of a small number (0.0001) is to ensure that the sum needs to be strictly greater than 0 to activate binary_var.")
print('-'*100)
print("(e) Write down the following constraint in mathematical notation: At least 30% of the invitations should go to players that can play the guard position (G, G/F).")
print("Some feature engineering made it possible to select only those players who can play the guard position. ")
print("Hence - sum ( 'guard' ) >= 0.3 *21 ")
print('-'*100)
print("(f) What is the optimal objective function value?")
print(round(model.ObjVal))
print('-'*100)

print("(g) How many guards (G, G/F) are invited to the training camp?")
model_run(21,1)
# Process the solution
selected_players = [i for i in players.index if x[i].X > 0.5]
selected_guard_players = [i for i in players.index if (x[i].X > 0.5) and (players.loc[i, 'guard'] == 1)]

print("Selected Guards:", selected_guard_players)
print("Selected Guards count:", len(selected_guard_players), "Percentage:", round(len(selected_guard_players)/21*100,2), "%")
print("Total Selected Players:", selected_players)
print("Total Selected Players count:", len(selected_players))

print('-'*100)
print("(h) What is the smallest number of training camp invitations that can be sent before the model yields an infeasible solution? What constraint(s) cannot be satisfied?")
invitations = 21
while model.status != gp.GRB.INFEASIBLE:
    try:
        model_run(invitations,0)
        if invitations == 0:
            break
        invitations = invitations - 1
        if model.status == gp.GRB.INFEASIBLE:
            break
    except:
        print("Error occurred at invitations :", invitations)
        print('-'*100)
        break
try :
    print("Invitations :", invitations+1)
    model_run(invitations+1,1)
except:
    print("--- ¯\_(ツ)_/¯ ---")
    
model.computeIIS()  # Compute IIS
print("The following constraint(s) cannot be satisfied:")
for c in model.getConstrs():
    if c.IISConstr:
        print(f"{c.constrName} is part of the IIS.")
for v in model.getVars():
    if v.IISLB > 0 or v.IISUB > 0:
        print(f"Variable bounds for {v.VarName} are part of the IIS.")




print('-'*100)
print("(i) Describe (do not implement) the challenge of modifying your solution approach to ensure that players with a total score of 12 or under would not be invited to training camp.")
print("we can add a new constraint to ensure that players with a total score of 12 or under would not be invited to training camp.")
print("excluding players from ",players[players[skills].sum(axis=1) <= 12].index)
print(" the constraints should also be modified Eg -  x[i] in (20,22,23,24),  which can be done using list matching and dropping identical")
print('-'*100)
print("(j) What do you perceive as a problem with Victor’s approach of choosing participants?")
print("1. Player number 1-10,11-20,etc., should not have correlation with their invite")
print("2. Olympics needs a 12 player team. If coach doesn’t have selection authority then choosing from a smaller 21 top pool is better")
print("3. Skills of the players should be considered. Like pool must have 3 on skill and match with people with 3 on other skill.")
print("4. Objective should be to Maximize each skill of the team or based on Opponent's skill weights can be assigned.")

print('-'*100)
print("thank you !")
print('-'*100)