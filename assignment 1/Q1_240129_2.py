
from gurobipy import GRB
import gurobipy as gb
import pandas as pd

# Load data
# Loading capacities 
capacity_direct_production = pd.read_csv('Capacity_for_Direct_Production_Facilities.csv')['Capacity'].tolist()
capacity_transship_distribution = pd.read_csv('Capacity_for_Transship_Distribution_Centers.csv')['Capacity'].tolist()
capacity_transship_production = pd.read_csv('Capacity_for_Transship_Production_Facilities.csv')['Capacity'].tolist()

# loading costs
cost_direct = pd.read_csv('Cost_Production_to_Refinement.csv').pivot(index='ProductionFacility', columns='RefinementCenter', values='Cost').values
cost_To_Transshipment = pd.read_csv('Cost_Production_To_Transshipment.csv').pivot(index='ProductionFacility', columns='TransshipmentHub', values='Cost').values
cost_From_Transshipment = pd.read_csv('Cost_Transshipment_to_Refinement.csv').pivot(index='TransshipmentHub', columns='RefinementCenter', values='Cost').values
demand = pd.read_csv('Refinement_Demand.csv')['Demand'].tolist()

# Model setup
model = gb.Model("Can2Oil Transshipment Problem")

# Constants
num_supply_nodes = 25
num_transshipment_production_facilities = 15
num_transshipment_hubs = 2
num_demand_locations = 5

# Decision variables
x = model.addVars(num_supply_nodes, num_demand_locations, lb=0, vtype=GRB.CONTINUOUS, name="Direct_Supply")
y = model.addVars(num_transshipment_production_facilities, num_transshipment_hubs, lb=0, vtype=GRB.CONTINUOUS, name="To_Transshipment")
z = model.addVars(num_transshipment_hubs, num_demand_locations, lb=0, vtype=GRB.CONTINUOUS, name="From_Transshipment")

# Objective function
model.setObjective(
    gb.quicksum(cost_direct[i][j] * x[i, j] for i in range(num_supply_nodes) for j in range(num_demand_locations)) +
    gb.quicksum(cost_To_Transshipment[i][k] * y[i, k] for i in range(num_transshipment_production_facilities) for k in range(num_transshipment_hubs)) +
    gb.quicksum(cost_From_Transshipment[k][j] * z[k, j] for k in range(num_transshipment_hubs) for j in range(num_demand_locations)),
    GRB.MINIMIZE
)

# Constraints
# Supply constraints
for i in range(num_supply_nodes):
    # Calculate the total supply from supply node i to all destinations
    total_supply = x.sum(i, '*')
    
    # Add a constraint to ensure that total supply from node i is within its capacity
    model.addConstr(total_supply <= capacity_direct_production[i])


# Transshipment supply constraints
for m in range(num_transshipment_production_facilities):
    # Calculate the total transshipment from production facility m to all hubs
    total_transshipment = y.sum(m, '*')

    # Add a constraint to ensure that total transshipment from facility i is within its capacity
    model.addConstr(total_transshipment <= capacity_transship_production[m])


# Transshipment constraints
for k in range(num_transshipment_hubs):
    # Calculate the total received from production facilities to hub k
    total_received = gb.quicksum(y[i, k] for i in range(num_transshipment_production_facilities))
    model.addConstr(total_received <= capacity_transship_distribution[k])
    # Calculate the total sent from hub k to all destinations
    total_sent = z.sum(k, '*')
    model.addConstr(total_sent <= capacity_transship_distribution[k])
    # Add a constraint to ensure that total_sent is less than or equal to total_received
    model.addConstr(total_sent == total_received)


# Demand constraints
for j in range(num_demand_locations):
    model.addConstr(x.sum('*', j) + z.sum('*', j) == demand[j])

# Solve the model
model.optimize()

# Results 
print("Total Transportation Cost: ", model.objVal)
for v in model.getVars():
    if v.X > 0:
        print(f"{v.VarName} = {v.X}")
        
print( "\n Printing out the solutions to the questions : ")
print('\n -------------------------------------------------------------------------------------')
print("\na) After solving the linear program, what is the optimal transportation cost? ")
if model.status == GRB.OPTIMAL:
    # Print the optimal objective value, which represents the minimum transportation cost
    optimal_cost = model.objVal
    print(f"Optimal Transportation Cost: {optimal_cost:.2f}")
else:
    print("No optimal solution found.")

print('\n -------------------------------------------------------------------------------------')

print('\nb) In the optimal solution, what proportion of canola oil is transshipped?')

total_canola_produced = sum(sum(x[i, j].x for j in range(num_demand_locations)) for i in range(num_supply_nodes))
total_canola_transshipped = sum(sum(y[i, k].x for k in range(num_transshipment_hubs)) for i in range(num_transshipment_production_facilities))
proportion_transshipped = total_canola_transshipped / (total_canola_produced+ total_canola_transshipped)

print(f"Proportion of Canola Oil Transshipped: {proportion_transshipped*100:.2f}%")

# Calculate the updated volume sent directly from supply nodes to demand locations
volume_direct = sum(x[i, j].X for i in range(num_supply_nodes) for j in range(num_demand_locations))
print("\n Volume Sent Directly:", volume_direct)

# Calculate the updated volume sent to transshipment hubs from production facilities
volume_transshipment = sum(y[i, k].X for i in range(num_transshipment_production_facilities) for k in range(num_transshipment_hubs))
print("\n Volume Sent to Transshipment Hubs:", volume_transshipment)

print('\n -------------------------------------------------------------------------------------')
print("\nc) The model does not currently limit that amount of canola oil that is transshipped. How would you modify the objective function to account for this? Formulate and solve this model.")
print(" by adding a new penality of 0.5 times the transportation cost to limit the amount of canola oil that can be transshipped. ")
model.setObjective(
    gb.quicksum(cost_direct[i][j] * x[i, j] for i in range(num_supply_nodes) for j in range(num_demand_locations)) +
    gb.quicksum(cost_To_Transshipment[i][k] * y[i, k] *1.5 for i in range(num_transshipment_production_facilities) for k in range(num_transshipment_hubs)) +
    gb.quicksum(cost_From_Transshipment[k][j] * z[k, j]*1.5 for k in range(num_transshipment_hubs) for j in range(num_demand_locations)),
    GRB.MINIMIZE
)

# Solve the model
model.optimize()
# Results
cost = (gb.quicksum(cost_direct[i][j] * x[i, j] for i in range(num_supply_nodes) for j in range(num_demand_locations)) +
    gb.quicksum(cost_To_Transshipment[i][k] * y[i, k] for i in range(num_transshipment_production_facilities) for k in range(num_transshipment_hubs)) +
    gb.quicksum(cost_From_Transshipment[k][j] * z[k, j] for k in range(num_transshipment_hubs) for j in range(num_demand_locations))
)
print("Total Transportation Cost: ", cost.getValue())
for v in model.getVars():
    if v.X > 0:
        print(f"{v.VarName} = {v.X}")

# Calculate the updated volume sent directly from supply nodes to demand locations
U_volume_direct = sum(x[i, j].X for i in range(num_supply_nodes) for j in range(num_demand_locations))
print("\n Updated Volume Sent Directly:", U_volume_direct)

# Calculate the updated volume sent to transshipment hubs from production facilities
U_volume_transshipment = sum(y[i, k].X for i in range(num_transshipment_production_facilities) for k in range(num_transshipment_hubs))
print("\n Updated Volume Sent to Transshipment Hubs:", U_volume_transshipment)
p_total_canola_produced = sum(sum(x[i, j].x for j in range(num_demand_locations)) for i in range(num_supply_nodes))
p_total_canola_transshipped = sum(sum(y[i, k].x for k in range(num_transshipment_hubs)) for i in range(num_transshipment_production_facilities))
p_proportion_transshipped = p_total_canola_transshipped / (p_total_canola_produced+ p_total_canola_transshipped)

print(f"Proportion of Canola Oil Transshipped: {p_proportion_transshipped*100:.2f}%")

print('\n -------------------------------------------------------------------------------------')
print("\nd) Instead of modifying the objective function, how would you modify the constraint set to reduce the proportion of canola oil that is transshipped? Formulate and solve this model.")
print('\n This can be done by adding a new constraint for transshipment limit to the total demand. Considering 24.88% of total demand as transshipment limit. ')


# Define a transshipment limit as 24.88% of the total demand (output from previous question)
transshipment_limit = round(sum(demand) * 0.248854262144821, 0)

# Add a new constraint for transshipment limit
model.addConstr(gb.quicksum(z[i, k] for i in range(num_transshipment_hubs) for k in range(num_demand_locations)) <= transshipment_limit, "Transshipment_Limit")
model.addConstr(gb.quicksum(y[i, k] for i in range(num_transshipment_production_facilities) for k in range(num_transshipment_hubs)) <= transshipment_limit, "Transshipment_Limit")


# Solve the model
# Modify the objective function to minimize the transshipment balance
model.setObjective(
    gb.quicksum(cost_direct[i][j] * x[i, j] for i in range(num_supply_nodes) for j in range(num_demand_locations)) +
    gb.quicksum(cost_To_Transshipment[i][k] * y[i, k] for i in range(num_transshipment_production_facilities) for k in range(num_transshipment_hubs)) +
    gb.quicksum(cost_From_Transshipment[k][j] * z[k, j] for k in range(num_transshipment_hubs) for j in range(num_demand_locations)),
    GRB.MINIMIZE
)
# Solve the modified model
model.optimize()
# Results
cost_2 = (gb.quicksum(cost_direct[i][j] * x[i, j] for i in range(num_supply_nodes) for j in range(num_demand_locations)) +
    gb.quicksum(cost_To_Transshipment[i][k] * y[i, k] for i in range(num_transshipment_production_facilities) for k in range(num_transshipment_hubs)) +
    gb.quicksum(cost_From_Transshipment[k][j] * z[k, j] for k in range(num_transshipment_hubs) for j in range(num_demand_locations))
)
print("Total Transportation Cost: ", cost_2.getValue())
for v in model.getVars():
    if v.X > 0:
        print(f"{v.VarName} = {v.X}")
# Calculate the updated volume sent directly from supply nodes to demand locations
updated_volume_direct = sum(x[i, j].X for i in range(num_supply_nodes) for j in range(num_demand_locations))
print("\nUpdated Volume Sent Directly:", updated_volume_direct)

# Calculate the updated volume sent to transshipment hubs from production facilities
updated_volume_transshipment = sum(y[i, k].X for i in range(num_transshipment_production_facilities) for k in range(num_transshipment_hubs))
print("\nUpdated Volume Sent to Transshipment Hubs:", updated_volume_transshipment)
p2_total_canola_produced = sum(sum(x[i, j].x for j in range(num_demand_locations)) for i in range(num_supply_nodes))
p2_total_canola_transshipped = sum(sum(y[i, k].x for k in range(num_transshipment_hubs)) for i in range(num_transshipment_production_facilities))
p2_proportion_transshipped = p2_total_canola_transshipped / (p2_total_canola_produced+ p2_total_canola_transshipped)

print(f"Proportion of Canola Oil Transshipped: {p2_proportion_transshipped*100:.2f}%")
print('\n -------------------------------------------------------------------------------------')
print('\ne) Which of the two modeling approaches would you recommend the company take to determine a transportation plan that reduces the amount of canola oil that is transshipped?')
print('\n I recommend using the approach that modifies the objective function. This approach includes a penalty function that can be controlled or made dynamic for further optimization.')
print('\n Working by limiting the total demand can cause issues in controlling cost. May be we need to limit the cost to 10% of total for transshipment which is possible only though setting objective function.')

print('\n -------------------------------------------------------------------------------------')
print('\nf) Re-shoring is the practice of transferring overseas business operations closer to the home country. Given its prevalence in today’s economy, how would you alter the original model to favor producers closer to North America? Formulate and solve this model.')
print("Overseas operations are being brought closer to the home country, meaning Canada and the U.S. Therefore, we're giving higher preference to supply nodes 1-10.")
for i in range(num_supply_nodes):
    # Calculate the total supply from supply node i to all destinations
    total_supply = x.sum(i, '*')
    if i <= 9: # We're giving higher preference to supply nodes 1-10 from canada & US
        # Add a constraint to maximise home supply. setting to the production capacity.
        model.addConstr(total_supply == capacity_direct_production[i])
    else:
        # Add a constraint to ensure that total supply from node i is within its capacity
        model.addConstr(total_supply <= capacity_direct_production[i])
model.setObjective(
    gb.quicksum(cost_direct[i][j] * x[i, j] for i in range(num_supply_nodes) for j in range(num_demand_locations)) +
    gb.quicksum(cost_To_Transshipment[i][k] * y[i, k] for i in range(num_transshipment_production_facilities) for k in range(num_transshipment_hubs)) +
    gb.quicksum(cost_From_Transshipment[k][j] * z[k, j] for k in range(num_transshipment_hubs) for j in range(num_demand_locations)),
    GRB.MINIMIZE
)
# Solve the modified model
model.optimize()
# Results
cost_3 = (gb.quicksum(cost_direct[i][j] * x[i, j] for i in range(num_supply_nodes) for j in range(num_demand_locations)) +
    gb.quicksum(cost_To_Transshipment[i][k] * y[i, k] for i in range(num_transshipment_production_facilities) for k in range(num_transshipment_hubs)) +
    gb.quicksum(cost_From_Transshipment[k][j] * z[k, j] for k in range(num_transshipment_hubs) for j in range(num_demand_locations))
)
print("Total Transportation Cost: ", cost_3.getValue())
for v in model.getVars():
    if v.X > 0:
        print(f"{v.VarName} = {v.X}")
# Calculate the updated volume sent directly from supply nodes to demand locations
updated_volume_direct = sum(x[i, j].X for i in range(num_supply_nodes) for j in range(num_demand_locations))
print("\nUpdated Volume Sent Directly:", updated_volume_direct)

# Calculate the updated volume sent to transshipment hubs from production facilities
updated_volume_transshipment = sum(y[i, k].X for i in range(num_transshipment_production_facilities) for k in range(num_transshipment_hubs))
print("\nUpdated Volume Sent to Transshipment Hubs:", updated_volume_transshipment)
p3_total_canola_produced = sum(sum(x[i, j].x for j in range(num_demand_locations)) for i in range(num_supply_nodes))
p3_total_canola_transshipped = sum(sum(y[i, k].x for k in range(num_transshipment_hubs)) for i in range(num_transshipment_production_facilities))
p3_proportion_transshipped = p3_total_canola_transshipped / (p3_total_canola_produced+ p3_total_canola_transshipped)

print(f"Proportion of Canola Oil Transshipped: {p3_proportion_transshipped*100:.2f}%")
print('\n -------------------------------------------------------------------------------------')
print('\ng) Do you expect the optimal solution to the re-shoring model to be similar to the optimal solution of the model that attempts to reduce transshipment? Why or why not?')
print('\n The optimal solution differs from the attempt to reduce transshipment, due to the differences in constraints.')
print('\n The reshoring model leads to higher transportation cost due to geographic preference to home locations set at max production capacity. So reshoring has been adjusted for constraints while primary model is adjusted for cheapest transport cost.')
print('\n -------------------------------------------------------------------------------------')