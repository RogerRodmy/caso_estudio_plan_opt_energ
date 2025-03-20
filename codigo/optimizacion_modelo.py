import pulp
import matplotlib.pyplot as plt
import numpy as np

# Parámetros escalados
E_panel = 1.5         # kWh/día por panel fotovoltaico
E_turbine = 15        # kWh/día por aerogenerador
inverter_capacity = 55000  # Capacidad máxima del inversor (kWh/día)
land_available = 140000    # Tierra disponible (m^2)
land_panel = 2           # m^2 requeridos por cada panel
land_turbine = 50        # m^2 requeridos por cada aerogenerador

# Máximos disponibles de equipos
panels_max = 30000
turbines_max = 3000

# Generar demandas
# Demandas desde 1000 hasta 50000 kWh/día en pasos de 1000
demands = list(range(1000, 50001, 1000))

# Listas para almacenar resultados
demands_list = []
panels_list = []
turbines_list = []
excess_list = []
total_gen_list = []
energy_panels_list = []
energy_turbines_list = []

# Bucle de optimización
for demand in demands:
    # Definición del problema de optimización (minimización del excedente)
    model = pulp.LpProblem("Hybrid_System_Optimization", pulp.LpMinimize)
    
    # Variables de decisión
    x = pulp.LpVariable("Panels", lowBound=0, upBound=panels_max, cat=pulp.LpInteger)
    y = pulp.LpVariable("Turbines", lowBound=0, upBound=turbines_max, cat=pulp.LpInteger)
    E_excedente = pulp.LpVariable("ExcessEnergy", lowBound=0, cat=pulp.LpContinuous)
    
    # Restricción 1: Balance de energía
    # La energía generada debe cubrir la demanda más el excedente.
    model += E_panel * x + E_turbine * y == demand + E_excedente, "Energy_Balance"
    
    # Restricción 2: Capacidad del inversor
    model += E_panel * x + E_turbine * y <= inverter_capacity, "Inverter_Capacity"
    
    # Restricción 3: Disponibilidad de tierra
    model += land_panel * x + land_turbine * y <= land_available, "Land_Availability"
    
    # Restricción 4: Relación técnica entre paneles y aerogeneradores (x >= 1.5 * y)
    model += x >= 1.5 * y, "Technical_Ratio"
    
    # Función objetivo: Minimizar la energía excedente
    model += E_excedente, "Minimize_Excess_Energy"
    
    # Resolver el modelo (sin imprimir mensajes)
    model.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # Verificar solución y almacenar resultados
    if pulp.LpStatus[model.status] == "Optimal":
        opt_x = pulp.value(x)
        opt_y = pulp.value(y)
        opt_excess = pulp.value(E_excedente)
        total_generation = E_panel * opt_x + E_turbine * opt_y
    else:
        opt_x = np.nan
        opt_y = np.nan
        opt_excess = np.nan
        total_generation = np.nan

    demands_list.append(demand)
    panels_list.append(opt_x)
    turbines_list.append(opt_y)
    excess_list.append(opt_excess)
    total_gen_list.append(total_generation)
    energy_panels_list.append(E_panel * opt_x)
    energy_turbines_list.append(E_turbine * opt_y)

# Gráfico 1: Barras agrupadas de equipos vs. demanda
demands_array = np.array(demands_list)
panels_array = np.array(panels_list)
turbines_array = np.array(turbines_list)

# Posiciones para las barras agrupadas
pos = np.arange(len(demands_array))
bar_width = 0.4

plt.figure(figsize=(14, 7))
plt.bar(pos - bar_width/2, panels_array, width=bar_width, color='blue', label='Paneles')
plt.bar(pos + bar_width/2, turbines_array, width=bar_width, color='orange', label='Aerogeneradores')

plt.xlabel("Demanda (kWh/día)")
plt.ylabel("Cantidad de Equipos")
plt.title("Cantidad de Paneles y Aerogeneradores vs. Demanda")
plt.xticks(pos, demands_array, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Gráfico 2: Curvas de Energía generada por tecnología vs. demanda
energy_panels_array = np.array(energy_panels_list)
energy_turbines_array = np.array(energy_turbines_list)
total_gen_array = np.array(total_gen_list)

plt.figure(figsize=(14, 7))
plt.plot(demands_array, energy_panels_array, marker='o', linestyle='-', color='blue', label='Energía de Paneles')
plt.plot(demands_array, energy_turbines_array, marker='o', linestyle='-', color='orange', label='Energía de Aerogeneradores')
plt.plot(demands_array, total_gen_array, marker='o', linestyle='--', color='green', label='Energía Total Generada')

plt.xlabel("Demanda (kWh/día)")
plt.ylabel("Energía Generada (kWh/día)")
plt.title("Energía Generada por Paneles y Aerogeneradores vs. Demanda")
plt.legend()
plt.tight_layout()
plt.show()
