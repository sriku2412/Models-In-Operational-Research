#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:09:53 2024

@author: Haoyu Fang
"""

from gurobipy import GRB
import gurobipy as gb

#Create the optimization model 
model = gb.Model("Sunnyshore Bay Primal")

#Decision Variables
B1 = model.addVar(lb=0, vtype=gb.GRB.CONTINUOUS, name="BorrowMay_OneMonth")
B2 = model.addVar(lb=0, vtype=gb.GRB.CONTINUOUS, name="BorrowMay_TwoMonth")
B3 = model.addVar(lb=0, vtype=gb.GRB.CONTINUOUS, name="BorrowMay_ThreeMonth")
B4 = model.addVar(lb=0, vtype=gb.GRB.CONTINUOUS, name="BorrowJune_OneMonth")
B5 = model.addVar(lb=0, vtype=gb.GRB.CONTINUOUS, name="BorrowJune_TwoMonth")
B6 = model.addVar(lb=0, vtype=gb.GRB.CONTINUOUS, name="BorrowJuly_OneMonth")
C = model.addVars(4, lb=0, vtype=GRB.CONTINUOUS, name="Wealth")

model.setObjective(C[3], gb.GRB.MAXIMIZE)

#Add Constraints
constraint1 = model.addConstr(C[0] == 140000 + 180000 - 300000 + B1 + B2 + B3, "Period 1 Constraint")
constraint2 = model.addConstr(C[0] >= 25000)
constraint3 = model.addConstr(B1 + B2 + B3 <= 250000, "Loan Constraint in May")
constraint4 = model.addConstr(C[1] == C[0] + 260000 - 400000 + B4 + B5 - 1.0175 * B1, "Period 2 Constraint")
constraint5 = model.addConstr(C[1] >= 27500)
constraint6 = model.addConstr(B4 + B5 <= 150000, "Loan Constraint in June")
constraint7 = model.addConstr(C[2] == C[1] + 420000 - 350000 + B6 - B2 * 1.0225 - B4 * 1.0175, "Period 3 Constraint")
constraint8 = model.addConstr(C[2] >= 35000)
constraint9 = model.addConstr(C[2] >= 0.65 * (C[0]+C[1]))
constraint10 = model.addConstr(B6 <= 350000, "Loan Constraint in July")
constraint11 = model.addConstr(C[3] == C[2] + 580000 - 200000 - B3 * 1.0275 - B5 * 1.0225 - B6 * 1.0175, "Period 4 Constraint")
constraint12 = model.addConstr(C[3] >= 18000)

# Optimally solve the problem
model.optimize()

# Check if the optimization was successful
if model.status == gb.GRB.OPTIMAL:
    # Get the optimal solution and objective value
    optimal_B1 = B1.x
    optimal_B2 = B2.x
    optimal_B3 = B3.x
    optimal_B4 = B4.x
    optimal_B5 = B5.x
    optimal_B6 = B6.x

    optimal_objective_value = model.objVal

    # Print the results
    print("Optimal Solution:")
    print(f"BorrowMay_OneMonth = {optimal_B1}")
    print(f"BorrowMay_TwoMonth = {optimal_B2}")
    print(f"BorrowMay_ThreeMonth= {optimal_B3}")
    print(f"BorrowJune_OneMonth = {optimal_B4}")
    print(f"BorrowJune_TwoMonth = {optimal_B5}")
    print(f"BorrowJuly_OneMonth = {optimal_B6}")
    print("Optimal Objective Value:")
    print(f"z = {optimal_objective_value}")
    
    # These should equal the optimal solution to the dual problem
    print("Shadow Prices: ", (constraint1.pi, constraint2.pi, constraint3.pi, constraint4.pi, constraint5.pi, constraint6.pi, constraint7.pi, constraint8.pi, constraint9.pi, constraint10.pi, constraint11.pi))
    print('Total amount repay to the bank over the entire season is ', optimal_B1 * 1.0175 + optimal_B2 * 1.0225 + optimal_B3 * 1.0275 + optimal_B4 * 1.0175 + optimal_B5 * 1.0225 + optimal_B6 * 1.0175)
else:
    print("No feasible solution found.")



