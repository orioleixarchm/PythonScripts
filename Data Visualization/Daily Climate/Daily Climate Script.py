#Loading Packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Loading data
path = 'C:/Users/oriol/OneDrive/Programació/Python/Kaggle/Daily Climate'
file1 = 'DailyDelhiClimateTrain'
file2 = 'DailyDelhiClimateTest'
filetype = 'csv'
data_train = pd.read_csv(f'{path}/{file1}.{filetype}')
data_test = pd.read_csv(f'{path}/{file2}.{filetype}')
print(data_train.head(),'\n')
print(data_test.head(),'\n')
data = pd.concat([data_train, data_test], axis=0)
print(data.head(),'\n')

#Transforming
data['date'] = pd.to_datetime(sorted(data['date']))
print(data['date'].value_counts().nlargest(5),'\n')
data.drop_duplicates(subset=['date'], inplace=True)
data.set_index('date', inplace=True)
print(f'Max date is {data.index.max()} and minimum is {data.index.min()}.','\n')
data_yearly = data.resample('YS').mean()
data_monthy = data.resample('MS').mean()

#Agreggates
data_sets = [data_yearly, data_monthy]
for datos in data_sets:
    for column in datos.columns:
        datos[f'Avg_{column}'] = datos[column].mean()
    print(datos.head(),'\n')

#Enrichening
extreme_bounds = {'max_temp':40,
                  'max_hum':100,
                  'max_windsp':119,
                  'max_press':1020,
                  'min_temp':-20,
                  'min_hum':20,
                  'min_press':950}

def alert_creation(datos, extreme_values, avg=False, perc=0.2):
    if avg==False:
        datos['Temp_Alert'] = np.where((datos['meantemp'] >= (extreme_values['max_temp'])) |
                                        (datos['meantemp'] < extreme_values['min_temp']),1,0)
        datos['Hum_Alert'] = np.where((datos['humidity'] >= extreme_values['max_hum']) |
                                        (datos['humidity'] < extreme_values['min_hum']),1,0)
        datos['Pres_Alert'] = np.where((datos['meanpressure'] >= extreme_values['max_press']) |
                                        (datos['meanpressure'] < extreme_values['min_press']),1,0)
        datos['WindSp_Alert'] = np.where((datos['wind_speed'] >= extreme_values['max_windsp']),1,0)

        datos['Critical_Alert'] = np.where((datos['Temp_Alert'] + datos['Hum_Alert']
                                            + datos['Pres_Alert'] + datos['WindSp_Alert'])>=3,1,0)
    else:
        datos['Temp_Alert'] = np.where((datos['meantemp'] >= datos['Avg_meantemp']*(1+perc)) |
                                        (datos['meantemp'] < datos['Avg_meantemp']*(1-perc)),1,0)
        datos['Hum_Alert'] = np.where((datos['humidity'] >= datos['Avg_humidity']*(1+perc)) |
                                        (datos['humidity'] < datos['Avg_humidity']*(1-perc)),1,0)
        datos['Pres_Alert'] = np.where((datos['meanpressure'] >= datos['Avg_meanpressure']*(1+perc)) |
                                        (datos['meanpressure'] < datos['Avg_meanpressure']*(1-perc)),1,0)
        datos['WindSp_Alert'] = np.where((datos['wind_speed'] >= datos['Avg_wind_speed']*(1+perc)),1,0)

        datos['Critical_Alert'] = np.where((datos['Temp_Alert'] + datos['Hum_Alert']
                                            + datos['Pres_Alert'] + datos['WindSp_Alert'])>=3,1,0)
    return(datos)

data_monthy = alert_creation(data_monthy, extreme_bounds,avg=True)
data_alerts = data_monthy.groupby(data_monthy.index.year.astype(int))[['Temp_Alert', 'Hum_Alert', 'Pres_Alert', 'WindSp_Alert', 'Critical_Alert']].mean()
print(data_alerts,'\n')

#Visualization
data_monthy['meantemp'].plot(label='Monthly')
data_yearly['meantemp'].plot(label='Yearly')
plt.title('Average temperature')
plt.xlabel('Month & Year')
plt.ylabel('Avg Temperature Cº')
plt.legend()
plt.show()

data_alerts.plot()
plt.title('Alerts & Unusual Values')
plt.xlabel('Year')
plt.ylabel('Percentage of unusual values (%)')
plt.xticks(ticks=[2013,2014,2015,2016,2017])
plt.legend(loc='upper left')
plt.show()
