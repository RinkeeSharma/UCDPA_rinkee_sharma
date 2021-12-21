import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

AREA_POPULATION = 'Area Population'
AVG_AREA_NUMBER_OF_BEDROOMS = 'Avg. Area Number of Bedrooms'
AVG_AREA_NUMBER_OF_ROOMS = 'Avg. Area Number of Rooms'
AVG__AREA_HOUSE_AGE = 'Avg. Area House Age'
AVG__AREA_INCOME = 'Avg. Area Income'
PRICE = 'Price'
USA_HOUSING_FILE_PATH = 'USA_Housing.csv'
COUNTRY_WISE_TEMPERATURE_FILE_PATH = 'climatechange/GlobalLandTemperaturesByCountry.csv'
GLOBAL_TEMPERATURES_CSV = "climatechange/GlobalTemperatures.csv"

def read_data(path):
    return pd.read_csv(path)


def shape_data(path):
    data = path.shape
    return data


def show_data(path):
    data_set = read_data(path)
    print(data_set)
    shaped_data_set = shape_data(data_set)
    described_data_set = data_set.describe(include="all")
    missing_values_data_set = data_set.isna().sum()
    replaced_missing_values_data_set = data_set.replace(np.nan, 0)
    dropped_data_set = replaced_missing_values_data_set.dropna()
    dropped_data_set.describe(include="all")
    dropped_data_set.head()
    dropped_data_set.info()
    dropped_data_set.describe()
    print(shaped_data_set)
    print(described_data_set)
    print(missing_values_data_set)
    print(replaced_missing_values_data_set)
    print(dropped_data_set)


def house_pricing_analysis_of_a_country(path):
    usa_housing_df = read_data(path)
    usa_housing_df.head()
    usa_housing_df.info()
    usa_housing_df.describe()
    sns.distplot(usa_housing_df[PRICE])
    sns.heatmap(usa_housing_df.corr(), annot=True, cmap='coolwarm')
    x = usa_housing_df[[AVG__AREA_INCOME, AVG__AREA_HOUSE_AGE, AVG_AREA_NUMBER_OF_ROOMS,
                        AVG_AREA_NUMBER_OF_BEDROOMS, AREA_POPULATION]]
    y = usa_housing_df[PRICE]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)
    linear_regression = LinearRegression()
    linear_regression.fit(x_train, y_train)
    print(linear_regression.intercept_)
    cdf = pd.DataFrame(linear_regression.coef_, x.columns, columns=['Coeff'])
    predictions = linear_regression.predict(x_test)
    plt.figure(figsize=(16, 8))
    plt.scatter(y_test, predictions)
    plt.show()
    plt.figure(figsize=(16, 8))

    sns.distplot((y_test - predictions), bins=50)
    plt.show()

    print('MAE:', metrics.mean_absolute_error(y_test, predictions))
    print('MSE:', metrics.mean_squared_error(y_test, predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

    data = make_blobs(n_samples=200, n_features=2,
                      centers=4, cluster_std=1.8, random_state=101)
    plt.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap='rainbow')

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(data[0])
    kmeans.cluster_centers_
    kmeans.labels_
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 6))
    ax1.set_title('K Means')
    ax1.scatter(data[0][:, 0], data[0][:, 1], c=kmeans.labels_, cmap='rainbow')
    ax2.set_title("Original")
    ax2.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap='rainbow')


def global_warming_analysis(file_path):
    global_land_temp_country_wise = read_data(file_path)
    glbl_temp_cntry_clean = global_land_temp_country_wise[~global_land_temp_country_wise['Country'].isin(
        ['Denmark', 'Antarctica', 'France', 'Europe', 'Netherlands', 'United Kingdom', 'Africa', 'South America'])]
    glbl_temp_cntry_clean = glbl_temp_cntry_clean.replace(
        ['Denmark (Europe)', 'France (Europe)', 'Netherlands (Europe)', 'United Kingdom (Europe)'],
        ['Denmark', 'France', 'Netherlands', 'United Kingdom'])
    # Average temperature for each country
    countries = np.unique(glbl_temp_cntry_clean['Country'])
    mean_temp_list = []
    for country in countries:
        mean_temp_list.append(
            glbl_temp_cntry_clean[glbl_temp_cntry_clean['Country'] == country]['AverageTemperature'].mean())
    # Creating figure using 'plotly' library
    data = [dict(type='choropleth', locations=countries, z=mean_temp_list, locationmode='country names', text=countries,
                 marker=dict(line=dict(color='rgb(0,0,0)', width=1)),
                 colorbar=dict(autotick=True, tickprefix='', title='# Mean\nTemperature,\n°C'))]
    layout = dict(title='Mean land temperature in countries',
                  geo=dict(showframe=False, showocean=True, oceancolor='rgb(0,255,255)',
                           projection=dict(type='orthographic', rotation=dict(lon=60, lat=10), ),
                           lonaxis=dict(showgrid=True, gridcolor='rgb(102, 102, 102)'),
                           lataxis=dict(showgrid=True, gridcolor='rgb(102, 102, 102)')), )
    fig = dict(data=data, layout=layout)
    py.plot(fig, validate=False, filename='worldmap')
    # 'seaborn' as well as 'matplotlib' library is used to create horizontal bars for each countries
    mean_temp_bar_graph, countries_bar_graph = (list(x) for x in
                                                zip(*sorted(zip(mean_temp_list, countries), reverse=True)))
    sns.set(font_scale=0.9)
    f, ax = plt.subplots(figsize=(4.5, 50))
    colors_cw = sns.color_palette('coolwarm', len(countries))
    sns.barplot(mean_temp_bar_graph, countries_bar_graph, palette=colors_cw[::-1])
    ax.set(xlabel='Mean temperature', title='Mean land temperature in countries')

    glbl_temp = pd.read_csv(GLOBAL_TEMPERATURES_CSV)
    # Extract the year from a date
    years = np.unique(glbl_temp['dt'].apply(lambda x: x[:4]))
    mean_temp_wrld = []
    mean_temp_wrld_uncertainty = []
    for year in years:
        mean_temp_wrld.append(
            glbl_temp[glbl_temp['dt'].apply(lambda x: x[:4]) == year]['LandAverageTemperature'].mean())
        mean_temp_wrld_uncertainty.append(
            glbl_temp[glbl_temp['dt'].apply(lambda x: x[:4]) == year]['LandAverageTemperatureUncertainty'].mean())
    trace_0 = go.Scatter(x=years, y=np.array(mean_temp_wrld) + np.array(mean_temp_wrld_uncertainty), fill=None,
                         mode='lines', name='Uncertainty top', line=dict(color='rgb(0, 255, 255)', ))
    trace_1 = go.Scatter(x=years, y=np.array(mean_temp_wrld) - np.array(mean_temp_wrld_uncertainty), fill='tonexty',
                         mode='lines', name='Uncertainty bot', line=dict(color='rgb(0, 255, 255)', ))
    trace_2 = go.Scatter(x=years, y=mean_temp_wrld, name='Mean Temperature', line=dict(color='rgb(199, 121, 093)', ))
    data = [trace_0, trace_1, trace_2]
    layout = go.Layout(xaxis=dict(title='year'), yaxis=dict(title='Mean Temperature, °C'),
                       title='Mean land temperature in world', showlegend=False)
    fig = go.Figure(data=data, layout=layout, file_path="world")
    py.plot(fig)
    continents = ['China', 'United States', 'Niger', 'Greenland', 'Australia', 'Italy']
    mean_temp_year_country = [[0] * len(years[70:]) for i in range(len(continents))]
    j = 0
    for country in continents:
        all_temp_country = glbl_temp_cntry_clean[glbl_temp_cntry_clean['Country'] == country]
        i = 0
        for year in years[70:]:
            mean_temp_year_country[j][i] = all_temp_country[all_temp_country['dt'].apply(lambda x: x[:4]) == year][
                'AverageTemperature'].mean()
            i += 1
        j += 1
    traces = []
    colors = ['rgb(0, 255, 255)', 'rgb(255, 0, 255)', 'rgb(0, 0, 0)', 'rgb(255, 0, 0)', 'rgb(0, 255, 0)',
              'rgb(0, 0, 255)']
    for i in range(len(continents)): traces.append(
        go.Scatter(x=years[70:], y=mean_temp_year_country[i], mode='lines', name=continents[i],
                   line=dict(color=colors[i]), ))
    layout = go.Layout(xaxis=dict(title='year'), yaxis=dict(title='Mean Temperature, °C'),
                       title='Mean land temperature on the continents', )
    fig = go.Figure(data=traces, layout=layout, file_path="continents")
    py.plot(fig)



if __name__ == "__main__":
    # printing and showing data description which is used for analysis further.
    # It will help in understanding the data better
    show_data(USA_HOUSING_FILE_PATH)
    # house pricing analysis using supervised and unsupervised learning
    house_pricing_analysis_of_a_country(USA_HOUSING_FILE_PATH)

    # printing and showing data description which is used for analysis further.
    # It will help in understanding the data better
    show_data(COUNTRY_WISE_TEMPERATURE_FILE_PATH)
    show_data(GLOBAL_TEMPERATURES_CSV)
    # global warming analysis country wise
    global_warming_analysis(COUNTRY_WISE_TEMPERATURE_FILE_PATH)