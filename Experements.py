from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

from Runner import rmsle, rmsle_cv
from Tools import seed

if __name__ == "__main__":
    y_actual = [[i] for i in range(20)]
    y_predict = y_actual.copy()

    error = rmsle(y_actual, y_predict)
    print(error)

    mean_error = mean_squared_error(y_actual, y_predict)
    print(mean_error)

    lasso = Lasso(alpha=0.0005, random_state=seed)
    error_cv = rmsle_cv(lasso, y_actual, y_predict)
    print("Lasso score: {:.4f} ({:.4f})".format(error_cv.mean(), error_cv.std()))
