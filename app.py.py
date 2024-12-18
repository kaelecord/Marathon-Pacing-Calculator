# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 04:29:45 2024

@author: kaele
"""

import streamlit as st
import json
import time
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random

#data = pd.read_csv("C:/Users/kaele/OneDrive/Documents/IUPUI/Fall 2024/T562 - Sports Analytics/Final Project/data/final_data.csv")
data = pd.read_csv("https://raw.githubusercontent.com/kaelecord/Marathon-Pacing-Calculator/refs/heads/main/data/final_data.csv")

def seconds_to_time(split_seconds):
    minutes, seconds = divmod(split_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours == 0:
        time = f'{minutes}:{seconds:02d}'
    else:
        time = f'{hours:d}:{minutes:02d}:{seconds:02d}'

    return time

def  time_to_seconds(time):
    split_time = list(map(int,time.split(':')))

    if len(split_time) == 2:
        time_sec = split_time[0]*60 + split_time[1]
    elif len(split_time) == 3:
        if split_time[0] >=10:
            time_sec = split_time[0]*60 + split_time[1]
        else:
            time_sec = split_time[0]*3600 + split_time[1]*60 + split_time[2]
    else:
        time_sec = -999
    return(time_sec)

def calculate_pace_per_mile(seconds, distance):
    if distance == "half":
        miles = 13.1
    elif distance == 'finish':
        miles = 26.2
    else:
        miles = distance/1.60934 # km to miles conversion

    pace_seconds_per_mile = int(round(seconds/miles,0))

    return(seconds_to_time(pace_seconds_per_mile))

def get_neighboring_age_groups(age_group):
  all_age_groups = sorted(list(data['age_group'].unique()))
  age_group_index = all_age_groups.index(age_group)
  neighboring_age_groups = [age_group]
  if (age_group_index > 0) & (age_group_index != len(all_age_groups)-1):
    neighboring_age_groups.append(all_age_groups[age_group_index - 1])
    neighboring_age_groups.append(all_age_groups[age_group_index + 1])
  elif age_group_index == 0:
    neighboring_age_groups.append(all_age_groups[age_group_index + 1])
  else:
    neighboring_age_groups.append(all_age_groups[age_group_index - 1])
  return neighboring_age_groups

def get_neighboring_finish_groups(finish_group):
  all_finish_groups = sorted(list(data['finish_group'].unique()))
  finish_group_index = all_finish_groups.index(finish_group)

  neighboring_finish_groups = [finish_group]
  if (finish_group_index > 0) & (finish_group_index != len(all_finish_groups)-1):
    neighboring_finish_groups.append(all_finish_groups[finish_group_index -1])
    neighboring_finish_groups.append(all_finish_groups[finish_group_index + 1])
  elif finish_group_index == 0:
    neighboring_finish_groups.append(all_finish_groups[finish_group_index + 1])
  else:
    neighboring_finish_groups.append(all_finish_groups[finish_group_index - 1])
  return neighboring_finish_groups

def get_new_splits(age_group, gender, finish_group):
  neighboring_age_groups = get_neighboring_age_groups(age_group)
  neighboring_finish_groups = get_neighboring_finish_groups(finish_group)
  data_80 = data[(data['age_group'] == age_group) & (data['gender'] == gender) & (data['finish_group'] == finish_group)]
  data_15 = data[(data['age_group'].isin(neighboring_age_groups)) & (data['gender'] == gender) & (data['finish_group'] == finish_group)]
  data_5 = data[data['finish_group'].isin(neighboring_finish_groups)]

  new_splits = []
  for i in range(1, 11, 1):
    weighted_avg = (data_80[f'split_{i}'].mean()*0.8) + (data_15[f'split_{i}'].mean()*0.15) + (data_5[f'split_{i}'].mean()*0.05)
    new_splits.append(int(round(weighted_avg,0)) if not pd.isna(weighted_avg) else 0)

  return new_splits

def get_new_splits_with_strategy(age_group, gender, finish_group):
  pacing_strategy = ['Positive Split', 'Even Split', 'Negative Split']
  neighboring_age_groups = get_neighboring_age_groups(age_group)
  neighboring_finish_groups = get_neighboring_finish_groups(finish_group)
  new_splits = {'Positive Split': [],
                'Even Split': [],
                'Negative Split': []}
  for strategy in pacing_strategy:
    data_subset = data[(data['pacing_strategy'] == strategy)]
    data_80 = data_subset[(data_subset['age_group'] == age_group) & (data_subset['gender'] == gender) & (data_subset['finish_group'] == finish_group)]
    data_15 = data_subset[(data_subset['age_group'].isin(neighboring_age_groups)) & (data_subset['gender'] == gender) & (data_subset['finish_group'] == finish_group)]
    data_5 = data_subset[data_subset['finish_group'].isin(neighboring_finish_groups)]
    for i in range(1, 11, 1):
      weighted_avg = (data_80[f'split_{i}'].mean()*0.8) + (data_15[f'split_{i}'].mean()*0.15) + (data_5[f'split_{i}'].mean()*0.05)
      new_splits[strategy].append(int(round(weighted_avg,0)) if not pd.isna(weighted_avg) else 0)

  return new_splits

# Genetic Algorithm Code
def get_min_max_time(finish_group):
  times = finish_group.split("-")
  min_time = time_to_seconds(times[0])
  max_time = time_to_seconds(times[1])
  return min_time, max_time

def get_split_pace(split_number, split_seconds):
  if split_number == 5:
    miles = 0.682 # 20k to half marathon
  elif split_number == 6:
    miles = 2.427 # half to 25k
  elif split_number == 10:
    miles = 1.364 # 40k to finish
  else:
    miles = 3.107
  return int(round(split_seconds/miles,0))

def get_split_penalty(split_1, split_1_number, split_2, split_2_number):
  split_1_pace = get_split_pace(split_1_number, split_1)
  split_2_pace = get_split_pace(split_2_number, split_2)

  penalty_boundary  = split_1_pace*0.03
  if abs(split_1_pace - split_2_pace) <= penalty_boundary:
    return 0
  else:
    return 25 #penalty

def get_individual_penalty(individual):
  penalty = 0
  for i in range(1,10,1):
    penalty += get_split_penalty(individual[i-1], i, individual[i], i+1)
  return penalty

def fitness(individual, finish_group):
  t_min, t_max, = get_min_max_time(finish_group)
  t_optimal = int(round((t_max + t_min)/2,0))
  total_time = sum(individual)
  penalty = get_individual_penalty(individual)
  if total_time < t_min or total_time > t_max:
      return  abs(total_time - t_optimal) + penalty # if solution out of desired range add 5 min penalty to solution
  else:
      return penalty  # solution can be negative. we want to minimize this. negative means its under optimal and within the time range.

def create_ga_df(age_group, gender, finish_group):
  neighboring_age_groups = get_neighboring_age_groups(age_group)
  neighboring_finish_groups = get_neighboring_finish_groups(finish_group)
  ga_df = data[(data['age_group'].isin(neighboring_age_groups)) & (data['gender'] == gender) & (data['finish_group'].isin(neighboring_finish_groups))]
  if len(ga_df) < 1000:
    ga_df =  data[(data['age_group'].isin(neighboring_age_groups)) & (data['finish_group'].isin(neighboring_finish_groups))]

  if len(ga_df) < 1000:
    ga_df = data[(data['finish_group'].isin(neighboring_finish_groups))]

  return ga_df

def create_individual():
  individual = [random.choice(ga_df[f'split_{i}'].values) for i in range(1,11,1)]
  return individual

def create_population(pop_size):
  return [create_individual() for _ in range(pop_size)]

def evaluate_population(population, finish_group):
  return [fitness(individual, finish_group) for individual in population]

def tournament_selection(population, finish_group, tournament_size=5):
    # Randomly select a subset of individuals from the population
    tournament_entrants = random.sample(population, tournament_size)

    # Evaluate the fitness of the individuals in the tournament
    tournament_fitness = [fitness(individual, finish_group) for individual in tournament_entrants]

    # Find the best individual in the tournament (highest fitness)
    best_individual_index = tournament_fitness.index(min(tournament_fitness))  # Min because we're minimizing the fitness
    best_individual = tournament_entrants[best_individual_index]

    return best_individual

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(individual, mutation_rate):
  for i in range(len(individual)):
    if random.random() < mutation_rate:
      individual[i] = random.choice(ga_df[f'split_{i+1}'].values)
  return individual

def genetic_algorithm(pop_size, generations, crossover_rate, mutation_rate, elitism_size, age_group, gender, finish_group):
  population = create_population(pop_size)
  for generation in range(generations):
    fitness_scores = evaluate_population(population, finish_group)
    best_individual = population[fitness_scores.index(min(fitness_scores))]
    #print(f"Generation {generation}: Best Fitness = {min(fitness_scores)}, Best Individual = {best_individual}")
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0])]
    
    # Apply elitism: keep the top `elitism_size` individuals
    new_population = sorted_population[:elitism_size]
    while len(new_population) < pop_size:
      parent1 = tournament_selection(population, finish_group)
      parent2 = tournament_selection(population, finish_group)

      if random.random() < crossover_rate:
        child1, child2 = crossover(parent1, parent2)
      else:
        child1, child2 = parent1, parent2

      child1 = mutate(child1, mutation_rate)
      child2 = mutate(child2, mutation_rate)

      new_population.extend([child1, child2])

    population = new_population[:pop_size] # replace old population with new one
  final_fitness_scores = evaluate_population(population, finish_group)
  best_individual = population[final_fitness_scores.index(min(final_fitness_scores))]
  return best_individual, min(final_fitness_scores)

def create_single_plot(individual):
  split_distance_labels = ['start', '5k', '10k', '15k', '20k', 'half', '25k', '30k', '35k', '40k', 'finish']
  x = [0, 3.1, 6.2, 9.3, 12.4, 13.1, 15.5, 18.6, 21.7, 24.8, 26.2]
  y = [get_split_pace(i, individual[i-1]) for i in range(1,11,1)] + [get_split_pace(10, individual[9])]

  split_pace_times = [seconds_to_time(split_pace) for split_pace in y]

  # Create the figure
  fig = go.Figure()

  # Loop through the points to create horizontal and vertical segments
  for i in range(len(x) - 1):
      # Horizontal segment from (x[i], y[i]) to (x[i+1], y[i])
      fig.add_trace(go.Scatter(
          x=[x[i], x[i+1]], 
          y=[y[i], y[i]], 
          mode='lines', 
          line=dict(color='blue'),
          showlegend=False
      ))

      # Vertical segment from (x[i+1], y[i]) to (x[i+1], y[i+1])
      fig.add_trace(go.Scatter(
          x=[x[i+1], x[i+1]], 
          y=[y[i], y[i+1]], 
          mode='lines', 
          line=dict(color='blue'),
          showlegend=False
      ))

  # Set layout and labels
  fig.update_layout(
      title="Optimal Marathon Pacing Strategy",
      xaxis_title="Split/Distance",
      yaxis_title="Pace (seconds per mile)",
      xaxis=dict(tickmode='array', tickvals=x, ticktext=split_distance_labels),
      yaxis=dict(tickmode='array', tickvals=y, ticktext=split_pace_times),
      template="plotly_white"
  )

  # Show the figure
  return fig

def create_full_plot(all_pacing_strategies):
  strategy_color_map = {'Positive Split': 'red',
                        'Even Split': 'black',
                        'Negative Split': 'green',
                        'General Suggestion': 'blue'}
  split_distance_labels = ['start', '5k', '10k', '15k', '20k', 'half', '25k', '30k', '35k', '40k', 'finish']
  x = [0, 3.1, 6.2, 9.3, 12.4, 13.1, 15.5, 18.6, 21.7, 24.8, 26.2]  

  # Create the figure
  fig = go.Figure()

  # Loop through the points to create horizontal and vertical segments
  if len(all_pacing_strategies) == 0:
      return "Sorry but there is not enough data for the chosen filters :cry:"
  else:
      y_all = []
      for strategy in all_pacing_strategies:
          y_strategy = [get_split_pace(i, all_pacing_strategies[f'{strategy}'][i-1]) for i in range(1,11,1)] + [get_split_pace(10, all_pacing_strategies[f'{strategy}'][9])]
          y_all.extend(y_strategy)
          split_pace_times = [seconds_to_time(split_pace) for split_pace in y_strategy]

          for i in range(len(x) - 1):
              # Horizontal segment from (x[i], y[i]) to (x[i+1], y[i])
              fig.add_trace(go.Scatter(
                  x=[x[i], x[i+1]], 
                  y=[y_strategy[i], y_strategy[i]],
                  name = f'{strategy}',
                  mode='lines', 
                  line=dict(color=strategy_color_map[strategy]),
                  hovertemplate=f"Split: {split_distance_labels[i]}<br>Pace: {split_pace_times[i]} / mile",
                  legendgroup=f'{strategy}',
                  showlegend=i==0
              ))
        
              # Vertical segment from (x[i+1], y[i]) to (x[i+1], y[i+1])
              fig.add_trace(go.Scatter(
                  x=[x[i+1], x[i+1]], 
                  y=[y_strategy[i], y_strategy[i+1]], 
                  name = f'{strategy}',
                  mode='lines', 
                  line=dict(color=strategy_color_map[strategy]),
                  hovertemplate=f"Split: {split_distance_labels[i]}<br>Pace: {split_pace_times[i]} / mile",
                  legendgroup=f'{strategy}',
                  showlegend=False
              ))

  unique_pace_values = list(sorted(set(y_all)))
  subset_pace_values = unique_pace_values[::3]  # Select every other pace value to avoid overlap

  # Convert the subset of y-values to time format for display on the y-axis
  subset_pace_times = [seconds_to_time(pace) for pace in subset_pace_values]
  
  # Set layout and labels
  fig.update_layout(
      title="Optimal Pacing Strategy by Pacing Profile",
      xaxis_title="Split/Distance",
      yaxis_title="Pace (seconds per mile)",
      xaxis=dict(tickmode='array', tickvals=x, ticktext=split_distance_labels, showgrid=True),
      yaxis=dict(tickmode='array', tickvals=subset_pace_values, ticktext=subset_pace_times, showgrid=False), # FIX THIS
      template="plotly_white"
  )

  # Show the figure
  return fig
  
def get_pacing_df_cumulative_pace(split_number, time):
  match split_number:
    case 1:
      return int(round((time/3.107),2))
    case 2:
      return int(round((time/6.214),2))
    case 3:
      return int(round((time/9.321),2))
    case 4:
      return int(round((time/12.427),2))
    case 5:
      return int(round((time/13.109),2))
    case 6:
      return int(round((time/15.534),2))
    case 7:
      return int(round((time/18.641),2))
    case 8:
      return int(round((time/21.748),2))
    case 9:
      return int(round((time/24.855),2))
    case 10:
      return int(round((time/26.219),2))

def create_pacing_df(individual):
  split_distance_labels = ['start-5k', '5k-10k', '10k-15k', '15k-20k', '20k-half', 'half-25k', '25k-30k', '30k-35k', '35k-40k', '40k-finish']
  split_time = [seconds_to_time(individual[i-1]) for i in range(1,11,1)]
  split_times_paces = [seconds_to_time(get_split_pace(i, individual[i-1])) for i in range(1,11,1)]
  cumulative_time = []
  cumulative_time_formatted = []
  cumulative_pace = []
  for i in range(10):
    cumulative_time.append(sum(individual[0:i+1]))
    cumulative_time_formatted.append(seconds_to_time(sum(individual[0:i+1])))
    cumulative_pace.append(seconds_to_time(get_pacing_df_cumulative_pace(i+1, cumulative_time[i])))
  
  return pd.DataFrame({'Split (km)': split_distance_labels, 'Split Time': split_time, 'Split Pace': split_times_paces, 'Cumulative Time': cumulative_time_formatted, 'Cumulative Pace': cumulative_pace})

##########################
###                    ###
### STREAMLIT APP CODE ###
###                    ###
##########################

st.set_page_config(layout="wide")

title = st.container()
base_options = st.container()
model_option = st.container()
chart_display = st.container()
output = st.container()

with title:
    st.title("Marathon Pacing Strategy Calculator")

with base_options:
    col_1b, col_2b, col_3b = st.columns(3)
    gender_choice = col_1b.selectbox('Gender:', options = list(data['gender'].unique()))
    age_choice = col_2b.selectbox('Age Group:', options = sorted(list(data['age_group'].unique())))
    finish_time_choice = col_3b.selectbox('Goal Finish Time:', options = sorted(list(data['finish_group'].unique())))
    
with model_option:
    col_1m, col_2m, col_3m = st.columns(3)
    model_choice = col_2m.selectbox('Model Choice:', options = ['Weighted Average', 'Genetic Algorithm'])
    
    if model_choice == 'Weighted Average':
        results = get_new_splits_with_strategy(age_choice, gender_choice, finish_time_choice)
        results['General Suggestion'] = get_new_splits(age_choice, gender_choice, finish_time_choice)
        results = {key: value for key, value in results.items() if not any(v == 0 for v in value)}

    else:
        with st.spinner("Running Genetic Algorithm..."):
            ga_df = create_ga_df(age_choice, gender_choice, finish_time_choice)

            results, result_fitness = genetic_algorithm(pop_size=250, generations=25, crossover_rate=0.5, mutation_rate=0.05,  elitism_size=2, age_group=age_choice, gender=gender_choice, finish_group=finish_time_choice)
        st.success("Evolution Complete!")   
    
with chart_display:
    if len(results) == 0:
        st.write("Not enough data that meets criteria to generate accurate response :cry:")
    elif model_choice == 'Weighted Average':
        st.plotly_chart(create_full_plot(results), use_container_width=True)

        

with output:
    if model_choice == 'Weighted Average':
        if len(results) == 0:
            st.write("**Suggestion:** Try to slightly modify age group or goal finish time! ")
        else:
            columns = st.columns(len(results))
    
            for idx, (strategy, data) in enumerate(results.items()):
                col = columns[idx]  # Get the column corresponding to the current pacing strategy
                col.subheader(f"{strategy} Details:")  # Display the strategy's name
                col.dataframe(create_pacing_df(data), use_container_width=True, hide_index=True)
    else:
        col_1o, col_2o = st.columns([2,1])
        
        col_1o.plotly_chart(create_single_plot(results), use_container_width=True)
        col_2o.subheader("Pacing Strategy Details")
        col_2o.dataframe(create_pacing_df(results))
        
        
        
        
        



