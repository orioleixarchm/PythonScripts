#Importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

#Getting data
path = r'C:\Users\User\OneDrive\Programació\Python\Kaggle\NBA_24_25'.replace('\\','/')
print(os.listdir(path),'\n')
data = pd.read_csv(f'{path}/database_24_25.csv')
data['Data'] = pd.to_datetime(data['Data'])
print(data.head(),'\n')

colnames = {
    "Player": "Name of the player.",
    "Tm": "Abbreviation of the player's team.",
    "Opp": "Abbreviation of the opposing team.",
    "Res": "Result of the game for the player's team.",
    "MP": "Minutes played",
    "FG": "Field goals made.",
    "FGA": "Field goal attempts.",
    "FG%": "Field goal percentage.",
    "3P": "3-point field goals made.",
    "3PA": "3-point field goal attempts.",
    "3P%": "3-point shooting percentage.",
    "FT": "Free throws made.",
    "FTA": "Free throw attempts.",
    "FT%": "Free throw percentage.",
    "ORB": "Offensive rebounds.",
    "DRB": "Defensive rebounds.",
    "TRB": "Total rebounds",
    "AST": "Assists",
    "STL": "Steals",
    "BLK": "Blocks.",
    "TOV": "Turnovers.",
    "PF": "Personal fouls.",
    "PTS": "Total points",
    "GmSc": "Game Score, a metric summarizing player performance for the game.",
    "Data": "Date of the game in YYYY-MM-DD format."
}

player_stats = data.drop(columns=['Tm','Opp','Res','Data']).groupby('Player').mean().reset_index()
team_stats = data.drop(columns=['Player','Opp','Res','Data']).groupby('Tm').mean()

#Radar plot
player_stats_top_5 = player_stats.nlargest(5,'GmSc')
player_stats_bottom_5 = player_stats.nsmallest(5,'GmSc')
stats = ["PTS","STL","AST","TRB","MP"]
description = [colnames[stat] for stat in stats]

player_stats_top_5_norm = player_stats_top_5[stats].apply(lambda x: x/x.max(), axis=0)
player_stats_bottom_5_norm = player_stats_bottom_5[stats].apply(lambda x: x/x.max(), axis=0)
player_stats_bottom_5_norm[player_stats_bottom_5_norm==0] = 0.05

angles = np.linspace(0, 2*np.pi, len(stats), endpoint=False).tolist()
angles += angles[:1]

fig_I, axes_I = plt.subplots(1,2, figsize=(6,6), subplot_kw=dict(polar=True))
fig_I.tight_layout()
for i, row in player_stats_top_5_norm.iterrows():
    values_I = row.tolist()
    values_I += values_I[:1]
    axes_I[0].plot(angles, values_I, label=player_stats_top_5.loc[i,'Player'])
    axes_I[0].fill(angles, values_I, alpha=0.1)
    axes_I[0].set_thetagrids(np.degrees(angles[:-1]),description)
    axes_I[0].set_title('Main Stats for most performant players', pad=20)
    axes_I[0].legend()
for i, row in player_stats_bottom_5_norm.iterrows():
    values_II = row.tolist()
    values_II += values_II[:1]
    axes_I[1].plot(angles, values_II, label=player_stats_bottom_5.loc[i,'Player'])
    axes_I[1].fill(angles, values_II, alpha=0.1)
    axes_I[1].set_thetagrids(np.degrees(angles[:-1]),description)
    axes_I[1].set_title('Main Stats for most least players',  pad=20)
    axes_I[1].legend()

fig_II, axes_II = plt.subplots()
fig_II.tight_layout()
corr = data.loc[:,["FG","3P","FT","ORB","DRB","AST","STL","BLK","PF","PTS","MP"]].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=axes_II, linewidth=0.5, edgecolor='black', linecolor='black')
axes_II.set_title('Correlation of Main Statistics')

top_5_player_data = data.loc[data['Player'].isin(player_stats_top_5['Player']),['Data','Player','PTS']].reset_index()
top_5_player_data_day = top_5_player_data.groupby(['Data','Player']).mean().reset_index(level='Player')
top_5_player_data_day_index = top_5_player_data_day.index
top_5_player_data_day.index = top_5_player_data_day.index.to_period('W')
top_5_player_data_day = top_5_player_data_day.sort_index()
data_date = data.groupby('Data').size()
data_date_month = data_date.index.to_period('W')

fig_III, axis_III = plt.subplots(2,1)
fig_III.tight_layout()
sns.lineplot(top_5_player_data_day, x=top_5_player_data_day_index, y='PTS', hue='Player', marker='o', markeredgecolor='black', ax=axis_III[0])
axis_III[0].set_xticks(top_5_player_data_day.index)
axis_III[0].set_xticklabels(top_5_player_data_day.index.strftime('%y-%m-%d'))
axis_III[0].set_xlabel(None)
axis_III[0].set_ylabel('Points')
axis_III[0].set_title('Top 5 Most performant players Points over Time ')
sns.lineplot(x=data_date.index, y=data_date.values, marker='o', markeredgecolor='black', ax=axis_III[1])
axis_III[1].set_xticks(data_date_month)
axis_III[1].set_xticklabels(data_date_month.strftime('%y-%m-%d'))
axis_III[1].set_xlabel(None)
axis_III[1].set_ylabel('Number of Games')
axis_III[1].set_title('Games Timeline')

fig_IV, axis_IV = plt.subplots(3,2)
axis_IV = axis_IV.flatten()
fig_IV.tight_layout()
stats = ["FG","FG%","3P","3P%","FT","FT%"]
colors = ['navy','skyblue','darkorange','peachpuff','crimson','lightcoral']
for ax, stat, color in zip(axis_IV,stats,colors):
    datos = data.groupby('Player')[[stat,"PTS"]].mean().reset_index(drop=False)
    datos = datos.loc[datos["PTS"]>=7,:]
    datos['Player'] = datos['Player'].apply(lambda x: x.split('-')[0] if '-' in x else x)
    sns.barplot(data=datos.nlargest(5, stat), x='Player', y=stat, color=color, edgecolor='black', linewidth=1, ax=ax)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_title(f'Top 5 players by {colnames[stat]}')

fig_V, axis_V = plt.subplots(2,2)
axis_V = axis_V.flatten()
fig_V.tight_layout()
stats = ["PTS","AST","TRB","STL"]
colors = ['blue','orange','steelblue','firebrick']
for ax, stat, color in zip(axis_V,stats,colors):
    datos = data.groupby('Player')[stat].mean().reset_index(drop=False)
    datos['Player'] = datos['Player'].apply(lambda x: x.split('-')[0] if '-' in x else x)
    sns.barplot(data=datos.nlargest(5, stat), x='Player', y=stat, color=color, edgecolor='black', linewidth=1, ax=ax)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_title(f'Top 5 players by {colnames[stat]}')

fig_VI, axis_VI = plt.subplots(4,1)
axis_VI = axis_VI.flatten()
fig_VI.tight_layout(pad=1)
sns.regplot(data, x="FGA", y="PTS", color='navy', ax=axis_VI[0])
sns.scatterplot(data, x="FGA", y="PTS", edgecolor='black', linewidths=0.9, ax=axis_VI[0])
axis_VI[0].set_ylabel('Total Points')
axis_VI[0].set_xlabel(None)
axis_VI[0].set_title('Impact of field shots tried on total points')

sns.regplot(data, x="3PA", y="PTS", color='tomato', ax=axis_VI[1])
sns.scatterplot(data, x="3PA", y="PTS", edgecolor='black', linewidths=0.9, ax=axis_VI[1])
axis_VI[1].set_ylabel('Total Points')
axis_VI[1].set_xlabel(None)
axis_VI[1].set_title('Impact of 3 Pointers tried on total points')

sns.regplot(data, x="FTA", y="PTS", color='darkorange', ax=axis_VI[2])
sns.scatterplot(data, x="FTA", y="PTS", edgecolor='black', linewidths=0.9, ax=axis_VI[2])
axis_VI[2].set_ylabel('Total Points')
axis_VI[2].set_xlabel(None)
axis_VI[2].set_title('Impact of Free throws tried on total points')

sns.regplot(data.loc[data['FGA']<=20,:], x="FGA", y="PTS", color='navy', ax=axis_VI[3])
sns.scatterplot(data.loc[data['FGA']<=20,:], x="FGA", y="PTS", edgecolor='black', linewidths=0.9, ax=axis_VI[3])
sns.regplot(data.loc[data['3PA']<=20,:], x="3PA", y="PTS", color='tomato', ax=axis_VI[3])
sns.scatterplot(data.loc[data['3PA']<=20,:], x="3PA", y="PTS", edgecolor='black', linewidths=0.9, ax=axis_VI[3])
sns.regplot(data.loc[data['FTA']<=20,:], x="FTA", y="PTS", color='darkorange', ax=axis_VI[3])
sns.scatterplot(data.loc[data['FTA']<=20,:], x="FTA", y="PTS", edgecolor='black', linewidths=0.9, ax=axis_VI[3])
axis_VI[3].set_ylabel('Total Points')
axis_VI[3].set_xlabel(None)
axis_VI[3].set_title('Impact of all kinds of shots tried on total points')
plt.show()