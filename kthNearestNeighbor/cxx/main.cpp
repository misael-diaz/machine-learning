/*

Machine Learning						June 16, 2023

source: main.cpp
author: @misael-diaz

Synopsis:
Implements the Kth Nearest Neighbor KNN Algorithm.
Uses the KNN Algorithm to construct a predictive model.

Copyright (c) 2023 Misael Diaz-Maldonado
This file is released under the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

References:
[0] A Koenig and B Moo, Accelerated C++ Practical Programming by Example
[1] JJ McConnell, Analysis of Algorithms, 2nd edition
[2] https://www.statlearning.com/resources-second-edition

*/

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <limits>

struct Data
{
  double X;
  double y;

  Data() : X(0), y(0) {}

  Data(double const X, double const y) : X(X), y(y) {}

  double dist(const Data& data)
  {
    double const x1 = this -> X;
    double const x2 = data.X;
    return ( (x1 - x2) * (x1 - x2) );
  }
};

void test_knn();

int main ()
{
  test_knn();
  return 0;
}


// gets a feature `X' and the target `y' from the Advertising dataset (from ref[2])
std::vector<Data> dataset ()
{
  std::ifstream in;
  std::string const fname = "datasets/Advertising.txt";
  in.open(fname, std::ios::in);

  std::vector<Data> dset;
  if ( in.is_open() )
  {
    double record;
    double TV;
    double radio;
    double newspaper;
    double sales;
    while (in >> record >> TV >> radio >> newspaper >> sales)
    {
      Data const data(TV, sales);
      dset.push_back(data);
      //std::cout << record << "\t" << TV << "\t" << sales << std::endl;
    }
    in.close();
  }

  return dset;
}

void test_knn ()
{
  // we need this predicate lambda function for sorting the dataset
  auto const pred = [](const Data& data1, const Data& data2) -> bool {
    return (data1.X < data2.X);
  };

  std::vector<Data> dset = dataset();
  std::sort(dset.begin(), dset.end(), pred);

  Data const target(128, 0);
  auto const div = std::lower_bound(dset.begin(), dset.end(), target, pred);
  int index = std::distance(dset.begin(), div);

  // stores the nearest neighbors for testing the implementation of the KNN algorithm
  std::vector<double> distances;
  for (auto& elem : dset)
  {
    double const dist = elem.dist(target);
    distances.push_back(dist);
  }

  std::sort(distances.begin(), distances.end());

  // finds the 1st nearest neighbor
  int i = (index - 1);
  int j = index;
  Data first(dset[i]);
  Data second(dset[j]);
  double d1 = first.dist(target);
  double d2 = second.dist(target);

  if (d2 < d1)
  {
    int larger = i;
    i = j;
    j = larger;

    double const d = d1;
    Data tmp(first);

    first = second;
    second = tmp;
    
    d1 = d2;
    d2 = d;
  }

  // gets the index of the next tree node
  index = (i < j)? (i - 1) : (i + 1);

  // initializes the error of the kth nearest neighbor
  double diff = (distances[0] - d1);

  int last = 0;
  // finds the 2nd, 3rd, 4th, etc. nearest neighbors dynamically by traversing the tree
  for (int k = 1; k != 200; ++k)
  {
    Data data(dset[index]);
    double d = data.dist(target);

    if (d < d2)
    {
      Data tmp(second);
      second = data;
      data = tmp;

      int larger = j;
      j = index;
      index = larger;
    }

    i = j;
    j = index;

    first = second;
    second = data;

    d1 = first.dist(target);
    d2 = second.dist(target);

    index = (i < j)? (i - 1) : (i + 1);

    // updates the error
    diff += (distances[k] - d1);

    // switches algorithm if the left partition is depleted:
    if (index < 0)
    {
      last = (k + 1);
      index = j;
      break;
    }
    
  }

  // if the left partition is depleted we continue with the right partition in order
  for (int k = last; k != 200; ++k)
  {
    Data data(dset[index]);
    double d = data.dist(target);
    diff += (distances[k] - d);
    ++index;
  }

  std::cout << "test[0]: ";
  if (diff != 0)
  {
    std::cout << "FAIL" << std::endl;
  }
  else
  {
    std::cout << "PASS" << std::endl;
  }

}
