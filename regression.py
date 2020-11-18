import requests
import pandas
import scipy
import numpy
import sys


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    # YOUR IMPLEMENTATION HERE
    data_str = response.text.split()
    areas = numpy.array(list(map(float, data_str[0].split(',')[1:])))
    prices = numpy.array(list(map(float, data_str[1].split(',')[1:])))
    #print(areas)
    #print(prices)

    amin, amax = numpy.min(areas), numpy.max(areas)
    pmin, pmax = numpy.min(prices), numpy.max(prices)
    
    areas = (areas-amin)/(amax-amin)
    prices = (prices-pmin)/(pmax-pmin)


    w_0 = numpy.random.random()
    w_1 = numpy.random.random()
    w_2 = numpy.random.random()
    #print(w_0, w_1)
    b = 1.0
    lrate = 0.5
    # Training
    EPOCH = 200
    for epoch in (range(EPOCH)):

        loss = (w_0 * areas + w_1 * (areas)**(0.5) + w_2 * (areas)**2 - prices)
        #print('Epoch: %d Loss %f' %(epoch, numpy.mean(loss**2)))
        
        grad_w_0 = numpy.mean(2*loss*areas)
        grad_w_1 = numpy.mean(2*loss*(areas**(0.5)))
        grad_w_2 = numpy.mean(2*loss*(areas**2))
        grad_b = numpy.mean(2*loss)

        w_0 = w_0 - lrate*grad_w_0
        w_1 = w_1 - lrate*grad_w_1
        w_2 = w_2 - lrate*grad_w_2
        b = b - lrate*grad_b

    area = (area-amin)/(amax-amin)
    price = (w_0 * area + w_1 * (area)**(0.5) + w_2 * (area)**2)
    price = price*(pmax-pmin) + pmin
    return price



if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
