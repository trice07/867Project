import numpy as np
import xgboost as xgb
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def test(loc, model_loc):
    data = dl.load_data(loc)
    model = dl.load_model(model_loc)
    predictions = model.predict(data)
    return predictions


def get_hitter_vec():
    return
def get_pitcher_vec():
    return



def process_data():
    X = []
    Y = []
    player_dict = {}
    pitcher_dict = {}
    for i in range(2):
        names = ["batters_latent.csv", "pitchers_latent.csv"]
        dicts = [player_dict, pitcher_dict]
        f_name = names[i]
        di = dicts[i]
        df = pd.read_csv(f_name)
        for _, row in df.iterrows():
            di[row["name"]] = row.values[2:]
    print("DONE")
    errs = set()
    for i in range(1, 3):
        print(i)
        loc = "Data2016-"+ str(i) + ".csv"
        train_data = np.loadtxt(loc, delimiter=",", dtype=str)
        for i in range(1, len(train_data)):
            _, _, h_name, date, result, p_name = train_data[i]
            ops = get_event_OPS(result)
            if ops is not None:
                if h_name not in player_dict:
                    errs.add(h_name)
                elif p_name not in pitcher_dict:
                    errs.add(p_name)
                else:
                    h_vec = player_dict[h_name]
                    p_vec = pitcher_dict[p_name]
                    vec = [str(date)] + [str(i) for i in h_vec] + [str(i) for i in p_vec] + [ops]
                    X.append(vec)
    print(errs)
    print(len(errs))
    print(len(X))
    return np.array(X, dtype=str)


def build_model(m_type):
    if m_type == "xgb":
        return xgb.XGBRegressor(n_estimators=200)


def train(loc, model_type, save_loc=None):
    data = np.load(loc)
    X = data[:, 1:-1].astype(np.float32)
    Y = data[:, -1].astype(np.float32)
    model = build_model(model_type)
    # model = pickle.load(open("prelimweights.sav", "rb"))
    train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size=0.2, random_state=42)
    # model.compile(loss='mean_squared_error', optimizer="adam")
    model.fit(train_features, train_labels)
    # history = model.fit(train_features, train_labels, epochs=num_epochs, batch_size=batch_size, validation_data=(test_features, test_labels), verbose=True)
    if save_loc is not None:
        pickle.dump(model, open(save_loc, "wb"))
        # model.save(save_loc)
    # if plot:
    #     plt.plot(history.history['loss'])
    #     plt.plot(history.history['val_loss'])
    #     plt.title('model loss')
    #     plt.ylabel('loss')
    #     plt.xlabel('epoch')
    #     plt.legend(['train', 'test'], loc='upper left')
    #     plt.show()
    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)
    return train_predictions, train_labels, test_predictions, test_labels


def get_event_OPS(target):
    event_dict = {'Sac Fly': None,
                 'Pop Out': 0,
                 'Groundout': 0,
                 'Lineout': 0,
                 'Strikeout': 0,
                 'Walk': 1,
                 'Flyout': 0,
                 'Hit By Pitch': 1,
                 'Forceout': 0,
                 'Single': 1,
                 'Double': 2,
                 'Bunt Pop Out': 0,
                 'Runner Out': None,
                 'Intent Walk': 1,
                 'Home Run': 4,
                 'Bunt Groundout': 0,
                 'Grounded Into DP': 0,
                 'Fielders Choice Out': 0,
                 'Triple': 3,
                 'Double Play': 0,
                 'Field Error': 0,
                 'Sac Bunt': None,
                 'Fielders Choice': 0,
                 'Batter Interference': None,
                 'Sac Fly DP': 0,
                 'Strikeout - DP': 0,
                 'Catcher Interference': None,
                 'Fan interference': None,
                 'Triple Play': 0,
                 'Sacrifice Bunt DP': 0,
                  "": None}
    return event_dict[target]


if __name__=="__main__":
    loc = "TrainingData.npy"
    model_type = "xgb"
    save_loc = "prelimweights.sav"
    train_predictions, train_labels, test_predictions, test_labels = train(loc, model_type, save_loc)
    # plt.scatter(train_predictions, train_labels)
    # plt.xlabel("Train Predictions")
    # plt.ylabel("Train Labels")
    # plt.title("Training Results")
    # plt.show()
    # plt.scatter(test_predictions, test_labels)
    # plt.xlabel("Test Predictions")
    # plt.ylabel("Test Labels")
    # plt.title("Test Results")
    # plt.show()
    errors = np.abs(train_predictions-train_labels)
    # plt.close()
    plt.hist(errors, bins=200)
    plt.show()
    print("MSE Train: ", np.mean((train_predictions-train_labels)**2))
    print("MSE Test: ", np.mean((test_predictions-test_labels)**2))
