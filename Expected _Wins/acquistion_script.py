# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 13:10:57 2019

@author: Matthew
"""

import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

def remove_space(word):
    x = str(word)
    return x.lstrip()

#assert "Python" in driver.title
#stat_categories = ['Wins','Team', 'G', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'opp_G', 'opp_MP', 'opp_FG', 'opp_FGA', 'opp_FG%', 'opp_3P', 'opp_3PA', 'opp_3P%', 'opp_2P', 'opp_2PA', 'opp_2P%', 'opp_FT', 'opp_FTA', 'opp_FT%', 'opp_ORB', 'opp_DRB', 'opp_TRB', 'opp_AST', 'opp_STL', 'opp_BLK', 'opp_TOV', 'opp_PF', 'opp_PTS']
#df = pd.DataFrame(columns = stat_categories, data = np.zeros([1,len(stat_categories)]))

df = pd.read_excel('team_data.xlsx',index_col = 0)

start = 1984
end = 1989

years = np.linspace(start,end,end-start+1,dtype=int)

teams = ['SAS','LAL','CLE','NYK','BOS','IND','PHO','HOU','MIL','PHI','DET','DEN','CHI','ATL','POR','GSW','UTA']


for year in years:
    for team in teams:
        driver = webdriver.Chrome(r"C:\Users\Matthew\Downloads\chromedriver_win32_test\chromedriver.exe")
        driver.get("https://www.basketball-reference.com/teams/{}/{}.html".format(team,str(year)))
        
        table_id = driver.find_element_by_id("team_misc")
        wins = float(table_id.find_elements_by_tag_name('td')[0].text)
            
        table_id = driver.find_element_by_id("team_and_opponent")
        row = table_id.find_elements_by_tag_name('tr')
        team_stats = row[1].find_elements_by_tag_name('td')
        opp_team_stats = row[5].find_elements_by_tag_name('td')
        header = row[0].find_elements_by_tag_name('th')
            
            
        header = header[1:len(header)]
        categories = ['Wins','Team']
        for category in header:
            categories.append(category.text)
        for category in header:
            categories.append('opp_'+category.text)
        
        stats = [wins,team+'_'+str(year)]    
            
        for stat in team_stats:
            stats.append(float(stat.text))
        for stat in opp_team_stats:
            stats.append(float(stat.text))
            
        stats = np.array(stats)
            
        df_temp = pd.DataFrame(columns = categories, data = [stats])
        df = pd.concat([df,df_temp],ignore_index=True)
        driver.close()
        
df.to_excel('team_data.xlsx')





