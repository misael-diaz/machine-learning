/*

Machine Learning						May 24, 2023

source: main.cpp
author: @misael-diaz

Synopsis:
Implements an Artificial Neural Network ANN to predict temperature data. The ANN
construction is based on reference [2]. The temperature dataset has been borrowed from
reference [3].

Copyright (c) 2023 Misael Diaz-Maldonado
This file is released under the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

References:
[0] A Koenig and B Moo, Accelerated C++ Practical Programming by Example
[1] JJ McConnell, Analysis of Algorithms, 2nd edition
[2] https://towardsdatascience.com/step-by-step-guide-to-building-your-own-neural-network-from-scratch-df64b1c5ab6e
[3] https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

*/

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <random>
#include <limits>
#include <string>
#include <tuple>
#include <cmath>

#define NUM_FEATURES 4
#define NUM_ITERATIONS 0x00100000
#define TOLERANCE 3.814697265625e-06
#define ALPHA 0.0009765625


std::vector<double> dataset();
std::vector<double> transpose(std::vector<double>);
std::vector<double> normalize(std::vector<double>);
std::tuple<std::vector<double>, std::vector<double>> split(std::vector<double>);
std::vector<double> get_features(std::vector<double>);
std::vector<double> get_target(std::vector<double>);
std::vector<double> init();

std::tuple<
  std::vector<double>,
  double,
  std::vector<double>
> optimize (std::vector<double> weights, double bias,
	    std::vector<double> features, std::vector<double> target);

std::vector<double> prediction (std::vector<double> weights, double bias,
				std::vector<double> features);

std::vector<double> error (std::vector<double> y, std::vector<double> y_pred);

void savetxt (std::string filename, std::vector<double> y, std::vector<double> y_pred);


int main ()
{
  std::vector<double> train, test;
  std::vector<double> dset = normalize( transpose( dataset() ) );
  std::tie(train, test) = split(dset);
  std::vector<double> X_train = get_features(train);
  std::vector<double> y_train = get_target(train);

  const std::vector<double>::size_type num_features = NUM_FEATURES;
  const std::vector<double>::size_type measurements = (X_train.size() / num_features);
  std::vector<double> weights = init();
  double bias = 0;

  std::vector<double> costs;
  std::tie(weights, bias, costs) = optimize(weights, bias, X_train, y_train);

  std::vector<double> y_pred = prediction(weights, bias, X_train);
  std::vector<double> errors = error(y_train, y_pred);
  savetxt("train-results.txt", y_train, y_pred);

  std::vector<double> X_test = get_features(test);
  std::vector<double> y_test = get_target(test);
  savetxt("test-results.txt", y_test, prediction(weights, bias, X_test) );

  std::cout << "train set size: " << train.size() << std::endl;
  std::cout << "test set size: " << test.size() << std::endl;

  /*
  for (auto error : errors)
  {
    std::cout << std::scientific << std::setprecision(12) << error << std::endl;
  }

  std::cout << "costs:" << std::endl;
  for (auto cost : costs)
  {
    std::cout << std::scientific << std::setprecision(12) << cost << std::endl;
  }

  std::cout << "weights:" << std::endl;
  for (auto weight : weights)
  {
    std::cout << std::scientific << std::setprecision(12) << weight << std::endl;
  }

  */

  return 0;
}


std::vector<double> dataset ()	// reads dataset (borrowed from ref[3]) into vector
{
  std::ifstream in;
  in.open("temps.txt", std::ios::in);

  std::vector<double> data;
  if ( in.is_open() )
  {
    double elem;
    while (in >> elem)
    {
      data.push_back(elem);
    }
    in.close();
  }

  return data;
}


std::vector<double> transpose (std::vector<double> dataset)
{
  const std::vector<double>::size_type columns = (NUM_FEATURES + 1);
  const std::vector<double>::size_type measurements = (dataset.size() / columns);

  std::vector<double> ret( dataset.size() );
  for (std::vector<double>::size_type i = 0; i != columns; ++i)
  {
    for (std::vector<double>::size_type j = 0; j != measurements; ++j)
    {
      ret[j + measurements * i] = dataset[i + columns * j];
    }
  }

  return ret;
}


double max_temperature (std::vector<double> dataset)
{
  const std::vector<double>::size_type columns = (NUM_FEATURES + 1);
  const std::vector<double>::size_type measurements = (dataset.size() / columns);

  std::vector<double> temps( dataset.begin() + measurements, dataset.end() );

  double max = -std::numeric_limits<double>::infinity();
  for (double temp : temps)
  {
    if (temp > max)
    {
      max = temp;
    }
  }

  return max;
}


std::vector<double> normalize (std::vector<double> dataset)	// normalizes dataset
{
  const std::vector<double>::size_type columns = (NUM_FEATURES + 1);
  const std::vector<double>::size_type measurements = (dataset.size() / columns);

  std::vector<double> norm( dataset.size() );
  for (std::vector<double>::size_type i = 0; i != measurements; ++i)
  {
    double month = dataset[i] / 12.0;
    norm[i] = month;
  }

  double max = max_temperature(dataset);
  for (std::vector<double>::size_type i = measurements; i != dataset.size(); ++i)
  {
    norm[i] = dataset[i] / max;
  }

  return norm;
}


std::tuple<std::vector<double>, std::vector<double>> split (std::vector<double> dataset)
{
  const std::vector<double>::size_type columns = (NUM_FEATURES + 1);
  const std::vector<double>::size_type measurements = (dataset.size() / columns);
  const std::vector<double>::size_type stride = 8;

  std::vector<double> mask(measurements, 0);
  for (std::vector<double>::size_type i = 0; i < measurements; i += stride)
  {
    mask[i] = 1;
  }

  std::vector<double> test;
  for (std::vector<double>::size_type i = 0; i != columns; ++i)
  {
    for (std::vector<double>::size_type j = 0; j != measurements; ++j)
    {
      if (mask[j] == 1)
      {
	test.push_back(dataset[j + measurements * i]);
      }
    }
  }

  std::vector<double> train;
  for (std::vector<double>::size_type i = 0; i != columns; ++i)
  {
    for (std::vector<double>::size_type j = 0; j != measurements; ++j)
    {
      if (mask[j] == 0)
      {
	train.push_back(dataset[j + measurements * i]);
      }
    }
  }

  return std::make_tuple(train, test);
}


std::vector<double> get_features (std::vector<double> dataset)
{
  std::vector<double> X;
  const std::vector<double>::size_type columns = (NUM_FEATURES + 1);
  const std::vector<double>::size_type measurements = (dataset.size() / columns);

  // adds the months
  for (std::vector<double>::size_type i = 0; i != measurements; ++i)
  {
    double month = dataset[i];
    X.push_back(month);
  }

  // adds the temperature one day previous
  for (std::vector<double>::size_type i = 0; i != measurements; ++i)
  {
    double temp_1 = dataset[i + measurements];
    X.push_back(temp_1);
  }

  // adds the temperature two days previous
  for (std::vector<double>::size_type i = 0; i != measurements; ++i)
  {
    double temp_2 = dataset[i + 2 * measurements];
    X.push_back(temp_2);
  }

  // adds the forecast temperature from NOAA
  for (std::vector<double>::size_type i = 0; i != measurements; ++i)
  {
    double forecast_noaa = dataset[i + 3 * measurements];
    X.push_back(forecast_noaa);
  }

  return X;
}


std::vector<double> get_target (std::vector<double> dataset)
{
  std::vector<double> y;
  const std::vector<double>::size_type columns = (NUM_FEATURES + 1);
  const std::vector<double>::size_type measurements = (dataset.size() / columns);

  // adds target, the actual temperature
  for (std::vector<double>::size_type i = 0; i != measurements; ++i)
  {
    double actual = dataset[i + 4 * measurements];
    y.push_back(actual);
  }

  return y;
}


std::vector<double> init ()
{
  const std::vector<double>::size_type num_features = NUM_FEATURES;

  std::random_device randev;
  std::default_random_engine engine( randev() );
  std::uniform_real_distribution<double> r(0, 1);

  std::vector<double> weights(num_features);
  for (std::vector<double>::size_type i = 0; i != num_features; ++i)
  {
    weights[i] = pow(2, -4) * r(engine);
  }

  return weights;
}


// std::vector<double> linear (std::vector<double> weights,
//			       std::vector<double> features,
//			       double bias)
// Synopsis:
// Obtains the linear hypothesis:
// 			Z = w * X + b,
// where `w' is the weight, `X' the features, and `b' the bias.
//
// Parameters:
// weights	vector of size NUM_FEATURES
// features	vector of size NUM_FEATURES * MEASUREMENTS
// bias		scalar
//
// Returns:
// Z		vector of size MEASUREMENTS


std::vector<double> linear (std::vector<double> weights,
			    std::vector<double> features,
			    double bias)
{
  const std::vector<double>& X = features;
  const std::vector<double>::size_type num_features = NUM_FEATURES;
  const std::vector<double>::size_type measurements = (features.size() / num_features);

  std::vector<double> Z(measurements);
  for (std::vector<double>::size_type i = 0; i != measurements; ++i)
  {
    Z[i] = bias;
    for (std::vector<double>::size_type j = 0; j != num_features; ++j)
    {
      Z[i] += (weights[j] * X[i + measurements * j]);
    }
  }

  return Z;
}


std::vector<double> sigmoid (std::vector<double> Z)
{
  std::vector<double> s( Z.size() );
  for (std::vector<double>::size_type i = 0; i != Z.size(); ++i)
  {
    s[i] = 1.0 / (exp(-Z[i]) + 1.0);
  }
  return s;
}


// double forward_propagation (std::vector<double> weights,
//			       double bias,
//			       std::vector<double> features,
//			       std::vector<double> target)
//
// Synopsis:
// Implements forward propagation.
//
// Parameters:
// weights	vector of size NUM_FEATURES
// bias		scalar
// features	vector of size MEASUREMENTS * NUM_FEATURES
// target	vector of size MEASUREMENTS
//
// Returns:
// cost		scalar, the objective function to be minimized


double forward_propagation (std::vector<double> weights,
			    double bias,
			    std::vector<double> features,
			    std::vector<double> target)
{
  const std::vector<double>& w = weights;
  const double b = bias;
  const std::vector<double>& X = features;
  const std::vector<double>& y = target;

  const std::vector<double>::size_type measurements = y.size();

  const std::vector<double>& Z = linear(w, X, b);
  const std::vector<double>& A = sigmoid(Z);

  double cost = 0.0;
  for (std::vector<double>::size_type i = 0; i != measurements; ++i)
  {
    cost += ( y[i] * log(A[i]) + (1.0 - y[i]) * log(1.0 - A[i]) );
  }
  cost /= ( (double) -measurements );

  return cost;
}


// std::tuple backward_propagation (std::vector<double> weights,
//			            double bias,
//			            std::vector<double> features,
//			            std::vector<double> target)
//
// Synopsis:
// Implements backward propagation.
//
// Parameters:
// weights	vector of size NUM_FEATURES
// bias		scalar
// features	vector of size MEASUREMENTS * NUM_FEATURES
// target	vector of size MEASUREMENTS
//
// Returns:
// tuple	contains the weight and bias gradients


std::tuple<
  std::vector<double>,
  double
> backward_propagation (std::vector<double> weights, double bias,
			std::vector<double> features, std::vector<double> target)
{
  const std::vector<double>& w = weights;
  const double b = bias;
  const std::vector<double>& X = features;
  const std::vector<double>& y = target;

  const std::vector<double>::size_type num_features = NUM_FEATURES;
  const std::vector<double>::size_type measurements = y.size();

  const std::vector<double>& Z = linear(w, X, b);
  const std::vector<double>& A = sigmoid(Z);

  std::vector<double> dweights(num_features);
  double mult = 1.0 / ( (double)  measurements );
  for (std::vector<double>::size_type i = 0; i != num_features; ++i)
  {
    dweights[i] = 0.0;
    for (std::vector<double>::size_type j = 0; j != measurements; ++j)
    {
      dweights[i] += mult * ( (A[j] - y[j]) * X[j + measurements * i] );
    }
  }

  double dbias = 0;
  for (std::vector<double>::size_type i = 0; i != measurements; ++i)
  {
    dbias += mult * (A[i] - y[i]);
  }

  return std::make_tuple(dweights, dbias);
}


// implements the steepest descent step to update either the weight or bias
std::vector<double> update (std::vector<double> vec, std::vector<double> grad)
{
  const double alpha = ALPHA;
  for (std::vector<double>::size_type i = 0; i != vec.size(); ++i)
  {
    vec[i] += (-alpha * grad[i]);
  }
  return vec;
}


// computes the vector modulus
double modulus (std::vector<double> vec)
{
  double mod = 0;
  for (std::vector<double>::size_type i = 0; i != vec.size(); ++i)
  {
    mod += (vec[i] * vec[i]);
  }

  return sqrt(mod);
}


// predicts the temperature, expects the optimal weight and bias as parameters
std::vector<double> prediction (std::vector<double> weights,  double bias,
				std::vector<double> features)
{
  const std::vector<double>& w = weights;
  const double b = bias;
  const std::vector<double>& X = features;

  const std::vector<double>& Z = linear(w, X, b);
  const std::vector<double>& A = sigmoid(Z);

  return A;
}


// computes the difference between the actual and predicted temperatures
std::vector<double> error (std::vector<double> y, std::vector<double> y_pred)
{
  std::vector<double> errors( y.size() );
  for (std::vector<double>::size_type i = 0; i != y.size(); ++i)
  {
    errors[i] = (y[i] - y_pred[i]);
  }
  return errors;
}


// std::tuple optimize (std::vector<double> weights,
//	 	        double bias,
//		        std::vector<double> features,
//		        std::vector<double> target)
//
// Synopsis:
// Attempts to optimize the hyper parameters of the Artificial Neural Network.
//
// Parameters:
// weights	vector of size NUM_FEATURES
// bias		scalar
// features	vector of size MEASUREMENTS * NUM_FEATURES
// target	vector of size MEASUREMENTS
//
// Returns:
// tuple	contains the (presumed) optimal weigth and bias and the cost


std::tuple<
  std::vector<double>,
  double,
  std::vector<double>
> optimize (std::vector<double> weights, double bias,
	    std::vector<double> features, std::vector<double> target)
{
  const double tol = TOLERANCE;
  const double alpha = ALPHA;

  std::vector<double>& w = weights;
  double b = bias;
  const std::vector<double>& X = features;
  const std::vector<double>& y = target;

  std::vector<double> costs;
  for (int i = 0; i != NUM_ITERATIONS; ++i)
  {
    double cost = forward_propagation(w, b, X, y);
    costs.push_back(cost);

    double db;
    std::vector<double> dw;
    std::tie(dw, db) = backward_propagation(w, b, X, y);

    std::vector<double> y_pred = prediction(w, b, X);
    std::vector<double> errors = error(y, y_pred);
    double mod = modulus(errors);

    if (mod < tol)
    {
      std::cout << "ANN has been optimized successfully" << std::endl;
      break;
    }

    w = update(w, dw);
    b += (-alpha * db);
  }

  return std::make_tuple(weights, bias, costs);
}


// saves the actual and the predicted temperatures to a plain text file
void savetxt (std::string filename, std::vector<double> y, std::vector<double> y_pred)
{
  std::ofstream out;
  out.open(filename, std::ios::out);
  for (std::vector<double>::size_type i = 0; i != y.size(); ++i)
  {
    out << std::scientific << std::setprecision(12)
	<< y[i] << " " << y_pred[i] << std::endl;
  }
  out.close();
}


// COMMENTS:
// There's room for improvement but it's not bad given it is the first time I attempt to
// build an Artificial Neural Network from scratch. The ANN model captures the qualitative
// behavior of the actual temperatures, for the ANN consists of just the input layer.
// You may wish to execute the python script plots.py to look at the graphs of the
// predicted temperature for the train and test sets.
//
// Decided to store the entire dataset in a vector rather than using a second-rank array
// because I prefer to use vectors over arrays in C++.
//
// And you will notice that the dataset does not contain all the features of the original
// (reference [3]) for simplicity.
//
// Matrix operations (transpose, multiplication, etc.) are implemented by taking into
// account the storage order, it is just a matter of playing with indexes.
//
// Could have passed vectors by reference to avoid the overhead of copying but since the
// dataset is ``small'' one can get away with that.
//
// Even though there are not as many comments, the code should be easy to follow if you
// are familiar with linear algebra, numerical optimization, and machine learning.


// TODO:
//
// [ ] consider passing vectors by reference to reduce the copy overhead
// [x] consider splitting into train and test sets to assess the ability of the ANN to
//     predict future values
// [x] consider adding a convergence criterion for optimize instead of just hoping that
//     the model has learned enough from the dataset upon reaching the MAX_ITERATIONS
