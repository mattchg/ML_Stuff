# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 19:16:53 2019

@author: Matthew
"""

import pandas as pd
import numpy as np

def get_team(title):
    return title.partition('_')[0]

def get_year(title):
    return title.partition('_')[2]

def convert_to_nba(key):
    df[key] = df[key]/(40/48)

df = pd.read_excel('college_team_stats.xlsx',index_col = 0)
df.head()
df = df.dropna(axis=1)

df.pop('PTS/G');
df.pop('opp_PTS/G');

team = df.pop('Team')
year = team.apply(func=get_year)
team = team.apply(func=get_team)

df['Team'] = team
df['Year'] = year

columns = ['FG', 'FGA', '2P', '2PA', '3P', '3PA',
       'FT', 'FTA', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS', 
        'opp_FG', 'opp_FGA', 'opp_TRB', 'opp_PTS']

convert_to_nba(columns)

df1 = pd.read_excel('team_data.xlsx',index_col = 0)
team = df1.pop('Team')
year = team.apply(func=get_year)
team = team.apply(func=get_team)
df1['Team'] = team
df1['Year'] = year

df1 = df1[df.columns]

test = pd.concat([df1,df],ignore_index=True)
test.to_excel('more_data.xlsx')
