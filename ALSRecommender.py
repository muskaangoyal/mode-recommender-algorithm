# Importing Libraries
import os
import json
import argparse
import time
import gc
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col, lower
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

class ALSRecommender:
# ALSRecommender utilised Alternating Least Square Matric Factorisation
# collaborative filtering to provide recommendations

    def __init__(self, spark_session, shopData, ratings):
            # shopData: The file containing collection of clothes we have in our database
            # ratings: The file containing user's ratings of the ShopData items
            self.spark = spark_session
            self.sc = spark_session.sparkContext
            self.shopDataDF = self._load_file(shopData).select(['productId', 'tags'])
            self.userDataDF = self.load_file(userData).select(['userId', 'preferences'])
            # Create a temporary ratings file based on tags and preferences which will contain userId, productId and rating
            self.ratingsDF = self.load_file(create_ratings_file(shopDataDF, userDataDF))
            self.model = ALS(
                userCol='userId',
                itemCol='productId',
                ratingCol='rating',
                coldStartStrategy="drop"

    def _load_file(self, filepath):
          # load json file into memory as spark DF
           return self.spark.read.load(filepath, format='json', header=True, inferSchema=True)

    def create_ratings_file(shopDataDF, userDataDF):
        """ Need help here to create a json file """
        #create empty json ratings file
        data = {}
        #keep adding ratings per (user, product) in the file
        userCol = userDataDF.select('userId')
        productCol = shopDataDF.select('productId')
        preference = userDataDF.select('preferences')
        tag = shopDataDF.select('tags')
        #Once number of users increase, we will have to find another way to create this file
        for (user: userCol):
            for (product: productCol):
                for (tag: tags):
                    #we can improve this rating later for sure
                    # if tag is equal to preference, then make the rating equal to 1
                    # if par of tag is equal to preference, then the rating is the same fraction
                    # otherwise zero
                    if ():
                        rating = 1
                    else():
                        rating = 0
                    #add user, product, rating in the data dictionary
                    #data[user] = {product, rating}
        ratings_data = json.dumps(data)
        return ratings_data

class Dataset:
    """
    data object make loading raw files easier
    """
    def __init__(self, spark_session, filepath):
        """
        spark dataset constructor
        """
        self.spark = spark_session
        self.sc = spark_session.sparkContext
        self.filepath = filepath
        # build spark data object
        self.RDD = self.load_file_as_RDD(self.filepath)
        self.DF = self.load_file_as_DF(self.filepath)

    def load_file_as_RDD(self, filepath):
        ratings_RDD = self.sc.textFile(filepath)
        header = ratings_RDD.take(1)[0]
        return ratings_RDD \
            .filter(lambda line: line != header) \
            .map(lambda line: line.split(",")) \
            .map(lambda tokens: (int(tokens[0]), int(tokens[1]), float(tokens[2]))) # noqa

    def load_file_as_DF(self, filepath):
        ratings_RDD = self.load_file_as_rdd(filepath)
        ratingsRDD = ratings_RDD.map(lambda tokens: Row(
            userId=int(tokens[0]), productId=int(tokens[1]), rating=float(tokens[2]))) # noqa
        return self.spark.createDataFrame(ratingsRDD)

def tune_ALS(model, train_data, validation_data, maxIter, regParams, ranks):
    """
    grid search function to select the best model based on RMSE of
    validation data
    Parameters
    ----------
    model: spark ML model, ALS
    train_data: spark DF with columns ['userId', 'productId', 'rating']
    validation_data: spark DF with columns ['userId', 'productId', 'rating']
    maxIter: int, max number of learning iterations
    regParams: list of float, one dimension of hyper-param tuning grid
    ranks: list of float, one dimension of hyper-param tuning grid
    Return
    ------
    The best fitted ALS model with lowest RMSE score on validation data
    """
    # initial
    min_error = float('inf')
    best_rank = -1
    best_regularization = 0
    best_model = None
    for rank in ranks:
        for reg in regParams:
            # get ALS model
            als = model.setMaxIter(maxIter).setRank(rank).setRegParam(reg)
            # train ALS model
            model = als.fit(train_data)
            # evaluate the model by computing the RMSE on the validation data
            predictions = model.transform(validation_data)
            evaluator = RegressionEvaluator(metricName="rmse",
                                            labelCol="rating",
                                            predictionCol="prediction")
            rmse = evaluator.evaluate(predictions)
            print('{} latent factors and regularization = {}: '
                  'validation RMSE is {}'.format(rank, reg, rmse))
            if rmse < min_error:
                min_error = rmse
                best_rank = rank
                best_regularization = reg
                best_model = model
    print('\nThe best model has {} latent factors and '
          'regularization = {}'.format(best_rank, best_regularization))
    return best_model

def parse_args():
    parser = argparse.ArgumentParser(
        prog="Products Recommender",
        description="Run ALS Product Recommender")
    parser.add_argument('--path', nargs='?', default='../data/productLens',
                        help='input data path')
    parser.add_argument('--shopData_filename', nargs='?', default='products.csv',
                        help='provide product filename')
    parser.add_argument('--ratings_filename', nargs='?', default='ratings.csv',
                        help='provide ratings filename')
    parser.add_argument('--product_name', nargs='?', default='',
                        help='provide your favoriate product name')
    parser.add_argument('--top_n', type=int, default=20,
                        help='top n product recommendations')
    return parser.parse_args()


if __name__ == '__main__':
    # get args
    args = parse_args()
    data_path = args.path
    shopData_filename = args.shopData_filename
    userData_filename = args.userData_filename
    shopData_name = args.shopData_name
    top_n = args.top_n
    # initial spark
    spark = SparkSession
        .builder
        .appName("products recommender")
        .getOrCreate()
    # initial recommender system
    recommender = AlsRecommender(
        spark,
        os.path.join(data_path, shopData_filename),
        os.path.join(data_path, ratings_filename))
    # set params
    recommender.set_model_params(10, 0.05, 20)
    # make recommendations
    recommender.make_recommendations(userId, top_n)
    # json function to print output
    # stop
    spark.stop()
