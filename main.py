import dataLoader as dl


def test(loc, model_loc):
    data = dl.load_data(loc)
    model = dl.load_model(model_loc)
    predictions = model.predict(data)
    return predictions


def train(loc, model_type, save_loc=None, plot=False):
    X, Y = dl.load_train_data(loc)
    model = dl.build_model(model_type)
    train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size=0.2, random_state=42)
    model.compile(loss='mean_squared_error', optimizer="adam")
    history = model.fit(train_features, train_labels, epochs=num_epochs, batch_size=batch_size, validation_data=(test_features, test_labels), verbose=True)
    if save_loc is not None:
        model.save(save_loc)
    if plot:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    test_predictions = model.predict(test_features)
    return history, test_predictions



