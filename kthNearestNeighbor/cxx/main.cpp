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

#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <numeric>
#include <vector>
#include <string>
#include <limits>

struct Data
{
  double X;
  double y;

  Data() : X(0), y(0) {}

  Data(double const X, double const y) : X(X), y(y) {}

  double dist(const Data& data) const
  {
    double const x1 = this -> X;
    double const x2 = data.X;
    return ( (x1 - x2) * (x1 - x2) );
  }
};

struct gData
{
  std::vector<double> X;	// features vector
  double y;			// response or output

  gData() : y(0) {}

  gData(const std::vector<double>& X, double const y)
  {
    this -> X = std::vector<double>(X);
    this -> y = y;
  }

  double dist(const gData& data) const
  {
    double d = 0;
    const std::vector<double>& x1 = this -> X;
    const std::vector<double>& x2 = data.X;
    for (std::vector<double>::size_type i = 0; i != X.size(); ++i)
    {
      d += (x1[i] - x2[i]) * (x1[i] - x2[i]);
    }

    return d;
  }
};

void ads();
void test();
void gtest();

int main ()
{
  ads();
  test();
  gtest();
  return 0;
}




std::vector<gData> gDataset ()
{
  std::ifstream in;
  std::string const fname = "datasets/Advertising.txt";
  in.open(fname, std::ios::in);

  std::vector<gData> dset;
  if ( in.is_open() )
  {
    double record;
    double TV;
    double radio;
    double newspaper;
    double sales;
    while (in >> record >> TV >> radio >> newspaper >> sales)
    {
      const std::vector<double> X{ TV, radio, newspaper };
      double const y = sales;
      gData const data(X, y);
      dset.push_back(data);
    }

    in.close();
  }

  return dset;
}


void isSorted (gData const& target, std::vector<gData> const& dset)
{
  auto const comp = [&target](const gData& data1, const gData& data2) -> bool {
    double const d1 = data1.dist(target);
    double const d2 = data2.dist(target);
    return (d1 < d2);
  };

  for (std::vector<gData>::size_type i = 0; i != (dset.size() - 1); ++i)
  {
    bool const isNextElemSmaller = comp(dset[i + 1], dset[i]);
    if (isNextElemSmaller)
    {
      std::string const err = "KNN(): expects a sorted dataset";
      throw std::invalid_argument(err);
    }
  }
}


void hasInvalidInput (int const Kth, gData const& target, std::vector<gData> const& dset)
{
  // warns user about invalid input
  int const size = dset.size();
  if (Kth < 1 || Kth > size)
  {
    std::string const err = "KNN(): Kth outside the valid arange [1, " +
			    std::to_string(1 + size) + ")";
    throw std::invalid_argument(err);
  }

  // could increase the size limit later if really needed
  int const maxSize = std::numeric_limits<int>::max() / 2;
  if (size > maxSize)
  {
    std::string const err = "KNN(): expects a dataset size less than or equal to " +
			    std::to_string(maxSize);
    throw std::invalid_argument(err);
  }

  isSorted(target, dset);
}


gData knn (int const Kth, gData const& target, std::vector<gData> const& dset)
{
  // we need this predicate lambda function for sorting the dataset
  auto const pred = [&target](const gData& data1, const gData& data2) -> bool {
    double const d1 = data1.dist(target);
    double const d2 = data2.dist(target);
    return (d1 < d2);
  };

  hasInvalidInput(Kth, target, dset);

  int const size = dset.size();
  // divides into left and right partitions
  auto const div = std::lower_bound(dset.begin(), dset.end(), target, pred);
  int index = std::distance(dset.begin(), div);

  if (index == 0)
  {
    // target is greater than or equal to min value in dataset, O(1) look up in right
    return dset[Kth - 1];
  }

  if (index == size)
  {
    // target is less than or equal to max value in dataset, O(1) look up in left
    return dset[size - Kth];
  }

  // traversal algorithm: finds the 1st nearest neighbor
  int i = (index - 1);
  int j = index;
  gData first(dset[i]);
  gData second(dset[j]);
  double d1 = first.dist(target);
  double d2 = second.dist(target);

  if (d2 < d1)
  {
    int larger = i;
    i = j;
    j = larger;

    double const d = d1;
    gData tmp(first);

    first = second;
    second = tmp;

    d1 = d2;
    d2 = d;
  }

  index = (i < j)? (i - 1) : (i + 1);

  if (Kth == 1)
  {
    return first;
  }

  if (index < 0)		// caters left partition depletion
  {
    index = j;
    index += (Kth - 2);
    return dset[index];
  }

  if (index == size)		// caters right partition depletion
  {
    index = j;
    index -= (Kth - 2);
    return dset[index];
  }

  // traversal algorithm: finds the 2nd, 3rd, 4th, etc. nearest neighbors dynamically

  int last = 1;
  bool forward = true;
  for (int k = 1; k != Kth; ++k)
  {
    gData data(dset[index]);
    double d = data.dist(target);

    if (d < d2)
    {
      gData tmp(second);
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

    // switches algorithm if the left partition is depleted:
    if (index < 0)
    {
      forward = true;
      last = (k + 1);
      index = j;
      break;
    }

    // switches algorithm if the right partition is depleted:
    if (index >= size)
    {
      forward = false;
      last = (k + 1);
      index = j;
      break;
    }

  }

  // returns the Kth nearest neighbor when the traversal algorithm succeeds:
  if (last == 1 || last == Kth)
  {
    // Note:
    // when last == Kth, this means the algorithm succeeded just in time since there were
    // no more elements in the left (or right) partition when the algorithm succeeded
    return first;
  }


  // gets the Kth nearest neighbor in the right partition, O(1), by computing its location
  if (forward)
  {
    index += (Kth - last - 1);
    first = dset[index];
    return first;
  }
  else
  {
    index -= (Kth - last - 1);
    first = dset[index];
    return first;
  }
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


void isSorted (std::vector<Data> const& dset)
{
  auto const comp = [](const Data& data1, const Data& data2) -> bool {
    return (data1.X < data2.X);
  };

  for (std::vector<Data>::size_type i = 0; i != (dset.size() - 1); ++i)
  {
    bool const isNextElemSmaller = comp(dset[i + 1], dset[i]);
    if (isNextElemSmaller)
    {
      std::string const err = "KNN(): expects a sorted dataset";
      throw std::invalid_argument(err);
    }
  }
}


void hasInvalidInput (int const Kth, std::vector<Data> const& dset)
{
  // warns user about invalid input
  int const size = dset.size();
  if (Kth < 1 || Kth > size)
  {
    std::string const err = "KNN(): Kth outside the valid arange [1, " +
			    std::to_string(1 + size) + ")";
    throw std::invalid_argument(err);
  }

  // could increase the size limit later if really needed
  int const maxSize = std::numeric_limits<int>::max() / 2;
  if (size > maxSize)
  {
    std::string const err = "KNN(): expects a dataset size less than or equal to " +
			    std::to_string(maxSize);
    throw std::invalid_argument(err);
  }

  isSorted(dset);
}


Data knn (int const Kth, Data const& target, std::vector<Data> const& dset)
{
  // we need this predicate lambda function for sorting the dataset
  auto const pred = [](const Data& data1, const Data& data2) -> bool {
    return (data1.X < data2.X);
  };

  hasInvalidInput(Kth, dset);

  int const size = dset.size();
  // divides into left and right partitions
  auto const div = std::lower_bound(dset.begin(), dset.end(), target, pred);
  int index = std::distance(dset.begin(), div);

  if (index == 0)
  {
    // target is greater than or equal to min value in dataset, O(1) look up in right
    return dset[Kth - 1];
  }

  if (index == size)
  {
    // target is less than or equal to max value in dataset, O(1) look up in left
    return dset[size - Kth];
  }

  // traversal algorithm: finds the 1st nearest neighbor
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

  index = (i < j)? (i - 1) : (i + 1);

  if (Kth == 1)
  {
    return first;
  }

  if (index < 0)		// caters left partition depletion
  {
    index = j;
    index += (Kth - 2);
    return dset[index];
  }

  if (index == size)		// caters right partition depletion
  {
    index = j;
    index -= (Kth - 2);
    return dset[index];
  }

  // traversal algorithm: finds the 2nd, 3rd, 4th, etc. nearest neighbors dynamically

  int last = 1;
  bool forward = true;
  for (int k = 1; k != Kth; ++k)
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

    // switches algorithm if the left partition is depleted:
    if (index < 0)
    {
      forward = true;
      last = (k + 1);
      index = j;
      break;
    }

    // switches algorithm if the right partition is depleted:
    if (index >= size)
    {
      forward = false;
      last = (k + 1);
      index = j;
      break;
    }

  }

  // returns the Kth nearest neighbor when the traversal algorithm succeeds:
  if (last == 1 || last == Kth)
  {
    // Note:
    // when last == Kth, this means the algorithm succeeded just in time since there were
    // no more elements in the left (or right) partition when the algorithm succeeded
    return first;
  }


  // gets the Kth nearest neighbor in the right partition, O(1), by computing its location
  if (forward)
  {
    index += (Kth - last - 1);
    first = dset[index];
    return first;
  }
  else
  {
    index -= (Kth - last - 1);
    first = dset[index];
    return first;
  }
}


extern "C" void KNN(int const K,
		    double const target,
		    int const size,
		    const double* dset,
		    double* kthElem)
{
  const double* X = dset;
  const double* y = (dset + size);

  std::vector<Data> vset;
  for (int i = 0; i != size; ++i)
  {
    Data const data(X[i], y[i]);
    vset.push_back(data);
  }

  Data const t(target, 0);
  Data data = knn(K, t, vset);
  kthElem[0] = data.X;
  kthElem[1] = data.y;
}


void gtest ()
{
  std::vector<gData> const dset = gDataset();

  double diff = 0;
  for (const auto& target : dset)
  {
    std::vector<double> distances;
    for (const auto& elem : dset)
    {
      double const dist = elem.dist(target);
      distances.push_back(dist);
    }

    std::sort(distances.begin(), distances.end());

    auto const pred = [&target](const gData& data1, const gData& data2) -> bool {
      double const d1 = data1.dist(target);
      double const d2 = data2.dist(target);
      return (d1 < d2);
    };

    std::vector<gData> set(dset);
    std::sort(set.begin(), set.end(), pred);

    for (std::vector<gData>::size_type k = 0; k != dset.size(); ++k)
    {
      int const Kth = (k + 1);
      gData const kthElem = knn(Kth, target, set);
      double const computed = kthElem.dist(target);
      double const expected = distances[k];
      diff += (computed - expected) * (computed - expected);
    }
  }

  std::cout << "knn-test: ";
  if (diff != 0)
  {
    std::cout << "FAIL" << std::endl;
  }
  else
  {
    std::cout << "PASS" << std::endl;
  }

}


// tests by looking all the kth nearest neighbors for each dataset element (target)
void test ()
{
  // we need this predicate lambda function for sorting the dataset
  auto const pred = [](const Data& data1, const Data& data2) -> bool {
    return (data1.X < data2.X);
  };

  std::vector<Data> dset = dataset();
  std::sort(dset.begin(), dset.end(), pred);

  double diff = 0;
  for (const auto& target : dset)
  {
    std::vector<double> distances;
    for (const auto& elem : dset)
    {
      double const dist = elem.dist(target);
      distances.push_back(dist);
    }

    std::sort(distances.begin(), distances.end());

    for (std::vector<Data>::size_type k = 0; k != dset.size(); ++k)
    {
      int const Kth = (k + 1);
      Data const kthElem = knn(Kth, target, dset);
      double const computed = kthElem.dist(target);
      double const expected = distances[k];
      diff += (computed - expected) * (computed - expected);
    }
  }

  std::cout << "knn-test: ";
  if (diff != 0)
  {
    std::cout << "FAIL" << std::endl;
  }
  else
  {
    std::cout << "PASS" << std::endl;
  }

}


void ads ()
{
  // we need this predicate lambda function for sorting the dataset
  auto const pred = [](const Data& data1, const Data& data2) -> bool {
    return (data1.X < data2.X);
  };

  std::vector<Data> dset = dataset();
  std::sort(dset.begin(), dset.end(), pred);

  double const x_min = dset[0].X;
  double const x_max = dset[dset.size() - 1].X;
  std::vector<double>::size_type N = 256;
  double const dx = (x_max - x_min) / ( (double) N );

  std::vector<double> x(N + 1);
  std::vector<double> ids(N + 1);
  std::iota(ids.begin(), ids.end(), 0);
  auto linspace = [&dx, &x_min](double const& idx) -> double {
    return (x_min + idx * dx);
  };
  std::transform(ids.begin(), ids.end(), x.begin(), linspace);

  double diff = 0;
  int const size = dset.size();
  for (int k = 0; k != size; ++k)
  {
    int const K = (k + 1);
    std::vector<double> y;
    // finds the 1st Nearest Neighbors
    for (auto const& elem : x)
    {
      Data const target(elem, 0);

      std::vector<double> distances;
      for (const auto& e : dset)
      {
	double const dist = e.dist(target);
	distances.push_back(dist);
      }

      std::sort(distances.begin(), distances.end());

      Data const data = knn(K, target, dset);
      double const value = data.y;
      double const computed = data.dist(target);
      double const expected = distances[K - 1];
      diff += (computed - expected) * (computed - expected);
      y.push_back(value);
    }

    std::ofstream out;
    std::string fname = "results/" + std::to_string(K) + "thNearestNeighbors.txt";
    out.open(fname, std::ios::out);

    if ( !out.is_open() )
    {
      std::cout << "IO Error: failed to open " + fname << std::endl;
      return;
    }

    for (std::vector<double>::size_type i = 0; i != x.size(); ++i)
    {
      out << x[i] << "\t" << y[i] << std::endl;
    }

    out.close();
  }

  std::cout << "knn-ads-test: ";
  if (diff != 0)
  {
    std::cout << "FAIL" << std::endl;
  }
  else
  {
    std::cout << "PASS" << std::endl;
  }
}


// TODO:
// [ ] throw exceptions on invalid inputs
