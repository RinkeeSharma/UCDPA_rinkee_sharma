import pandas as pd

def import_data(filepath):
    data = pd.read_csv(filepath)
    return data

def analyse_data(filepath):
    data = import_data(filepath)
    analysed_data = data.isna()
    return analysed_data

if __name__ == "__main__":
    dataset = import_data("netflix_titles.csv")

    for i in dataset.iterrows():
        break;
    #print(missing_values)
    #print(dataset.info)
    #print(dataset)