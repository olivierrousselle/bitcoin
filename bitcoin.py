
""" https://coincodex.com/crypto/bitcoin/historical-data/ """


import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("bitcoin_2010-07-17_2024-06-08.csv", delimiter=",")
df['date'] = pd.to_datetime(df['End'], format='%Y-%m-%d')
df['year'] = df['date'].dt.year
df = df.sort_values(by='date')
df = df.set_index(df['date'])
del df['Start'], df['End'], df['date'], df['Open'], df['High'], df['Low']

all_dates = pd.date_range(start='2009-01-01', end='2010-07-17')
missing_dates_df = pd.DataFrame(index=all_dates, columns=df.columns)
df = pd.concat([missing_dates_df, df])
df['time'] = range(len(df))

df = df['2010-07-18':]


df['Close'].plot(figsize=(14, 7), color='black')
plt.yscale('log')
plt.xlabel('Date')
plt.ylabel('Price Bitcoin $ (log scale)')
plt.grid()
plt.show()

dates_halving = ['2012-11-28', '2016-07-09', '2020-05-11', '2024-04-20']
dates_ath = ['2011-06-09', '2013-12-05', '2017-12-17', '2021-11-09']
dates_low = ['2011-11-19', '2015-08-15', '2018-12-15', '2022-11-10']


df_ath = df.loc[dates_ath]
df_low = df.loc[dates_low ]

""" ----- Relation Price-Time in scale log-linear ----- """

X = df['time']
y = np.log(df['Close'])
X_ath = df_ath['time']
y_ath = np.log(df_ath['Close'])
X_low = df_low['time']
y_low = np.log(df_low['Close'])


def log_law(x, a, b, x0):
    return a * np.log(x-x0) + b

popt, _ = curve_fit(log_law, X, y, maxfev=10000)
y_pred = log_law(X, *popt)
r2 = r2_score(y, y_pred)
print("R² model:", round(r2,3))
popt_ath, _ = curve_fit(log_law, X_ath, y_ath, maxfev=10000)
y_pred_ath = log_law(X, *popt_ath)
r2_ath = r2_score(y_ath, log_law(X_ath, *popt_ath))
print("R² model ATH:", round(r2_ath,3))
popt_low, _ = curve_fit(log_law, X_low, y_low, maxfev=10000)
y_pred_low = log_law(X, *popt_low)
r2_low = r2_score(y_low, log_law(X_low, *popt_low))
print("R² model Low:", round(r2_low,3))

print(f"Model: P = {format(np.exp(popt[1]),'.2e')} (t-{round(popt[2],2)})^{round(popt[0],3)}")
print(f"Model ATH: P = {format(np.exp(popt_ath[1]),'.2e')} (t-{round(popt_ath[2],2)})^{round(popt_ath[0],3)}")
print(f"Model Low: P = {format(np.exp(popt_low[1]),'.2e')} (t-{round(popt_low[2],2)})^{round(popt_low[0],3)}")


plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Close'], label='Price data', color='black')
plt.plot(df.index, np.exp(y_pred), label=f'Power model (R² = {r2:.3f})', color='green')
plt.plot(df.index, np.exp(y_pred_ath), label=f'Power model ATH (R² = {r2_ath:.3f})', color='red')
plt.plot(df.index, np.exp(y_pred_low), label=f'Power model Low (R² = {r2_low:.3f})', color='purple')
#plt.plot(df.index, y_pred_log, label=f'Modèle logarithmique (R² = {r2_log:.2f})', color='green')
plt.xlabel('Date')
plt.ylabel('Price of Bitcoin (log scale)')
plt.legend()
plt.grid()
plt.yscale("log")
plt.show()



startDate = pd.to_datetime('2009-01-01', format='%Y-%m-%d')

"""P_target = 200000
t_target = ((np.log(P_target)-popt[2])/popt[0])**(1/popt[1])
targetDate = startDate + pd.Timedelta(days=int(t_target))
print("Date:", targetDate.date())"""

targetdate = '2025-12-01'
targetDate = pd.to_datetime(targetdate, format='%Y-%m-%d')
t_target = (targetDate-startDate).days
P_target = int(np.exp(log_law(t_target, *popt)))
P_target_ath = int(np.exp(log_law(t_target, *popt_ath)))
P_target_low = int(np.exp(log_law(t_target, *popt_low)))
print(f"On {targetdate}, the price will almost certainly be between {P_target_low} and {P_target_ath}, with a mean of {P_target}.")

date_range = pd.date_range(start=df.index[0], end=targetDate, freq='D')

plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Close'], label='Price data', color='black')
plt.plot(date_range, np.exp(log_law(np.arange(len(date_range))+df['time'].iloc[0], *popt)), label=f'Power model (R² = {r2:.3f})', color='green')
plt.plot(date_range, np.exp(log_law(np.arange(len(date_range))+df['time'].iloc[0], *popt_ath)), label=f'Power model ATH (R² = {r2_ath:.3f})', color='red')
plt.plot(date_range, np.exp(log_law(np.arange(len(date_range))+df['time'].iloc[0], *popt_low)), label=f'Power model Low (R² = {r2_low:.3f})', color='purple')
#plt.plot(df.index, y_pred_log, label=f'Modèle logarithmique (R² = {r2_log:.2f})', color='green')
plt.xlabel('Date')
plt.ylabel('Price of Bitcoin (log scale)')
plt.axvline(targetDate, linestyle='--', color='blue')
plt.scatter(targetDate, P_target_ath, color='red')
plt.scatter(targetDate, P_target_low, color='purple')
plt.legend()
plt.grid()
plt.yscale("log")
plt.show()


""" ----- Relation Price-Time in scale log-log ----- """


X = np.log(df['time']-popt[2])/np.log(10)
y = np.log(df['Close'])/np.log(10)
X_ath = np.log(df_ath['time']-popt[2])/np.log(10)
y_ath = np.log(df_ath['Close'])/np.log(10)
X_low = np.log(df_low['time']-popt[2])/np.log(10)
y_low = np.log(df_low['Close'])/np.log(10)

def linear_law(x, a, b):
    return a * x + b

popt, _ = curve_fit(linear_law, X, y, maxfev=10000)
y_pred = linear_law(X, *popt)
r2 = r2_score(y, y_pred)
popt_ath, _ = curve_fit(linear_law, X_ath, y_ath, maxfev=10000)
y_pred_ath = linear_law(X, *popt_ath)
r2_ath = r2_score(y_ath, linear_law(X_ath, *popt_ath))
popt_low, _ = curve_fit(linear_law, X_low, y_low, maxfev=10000)
y_pred_low = linear_law(X, *popt_low)
r2_low = r2_score(y_low, linear_law(X_low, *popt_low))

plt.figure(figsize=(14, 7))
plt.plot(X, y)
plt.plot(X, y_pred, label=f'Linear model (R² = {r2:.3f})', color='green')
plt.plot(X, y_pred_ath, label=f'Linear model ATH (R² = {r2_ath:.3f})', color='red')
plt.plot(X, y_pred_low, label=f'Linear model Low (R² = {r2_low:.3f})', color='purple')
plt.xlabel('$log_{10}$ (nb days)')
plt.ylabel('$log_{10}$ (price Bitcoin)')
plt.grid()
plt.legend()
plt.show()


""" ----- Relation Halving - ATH ----- """


df_halving = df.loc[dates_halving]
df_ath = df.loc[dates_ath][1:]

X = df_halving[:-1]['time']
y = df_ath['time']

popt, _ = curve_fit(linear_law, X, y, maxfev=10000)
y_pred = linear_law(X, *popt)
r2 = r2_score(y, y_pred)
#print("R²:", round(r2,3))

y_pred_dates = [startDate + pd.Timedelta(days=y_pred_) for y_pred_ in y_pred]
y_pred_ = linear_law(df_halving.iloc[-1]['time'], *popt)
y_pred_date_ = startDate + pd.Timedelta(days=y_pred_)
y_pred_dates.append(y_pred_date_)
print("Date next ATH:", y_pred_date_.date())

plt.figure(figsize=(14, 7))
plt.scatter(df_halving.index[:-1], df_ath.index, color='black', label='Data')
plt.scatter(df_halving.index[-1], y_pred_date_, color='red', label='Prediction')
plt.plot(df_halving.index, y_pred_dates, '--', label=f'Linear model (R² = {r2:.3f})')
plt.grid()
plt.legend()
plt.xlabel("Date halving")
plt.ylabel("Date ATH")
plt.show()
