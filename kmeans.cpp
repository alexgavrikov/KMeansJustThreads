/*
 * kmeans.cpp
 *
 *  Created on: 11 мая 2016 г.
 *      Author: user
 */

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <vector>
#include <thread>
#include "Common/read_write.h"

typedef std::vector<double> Point;
typedef std::vector<Point> Points;

// Gives random number in range [0..max_value]
size_t UniformRandom(const size_t max_value);

double Distance(const Point& point1, const Point& point2);
size_t FindNearestCentroid(const Points& centroids, const Point& point);

// Calculates new centroid position as mean of positions of 3 random centroids
Point GetRandomPosition(const Points& centroids);

void FunctionForThreads(const size_t left_loop_bound,
                        const size_t right_loop_bound,
                        const size_t K,
                        const size_t dimensions,
                        const Points& centroids,
                        const Points& data,
                        Points& centroids_new,
                        std::vector<size_t>& clusters,
                        std::vector<size_t>& clusters_sizes,
                        bool& converged);

Points KMeans(const Points& data, size_t K);

int main(int argc, char** argv) {
  if (argc != 4) {
    std::printf("Usage: %s number_of_clusters input_file output_file\n", argv[0]);
    return 1;
  }

  char* input_file = argv[2];
  std::ifstream input;
  input.open(input_file, std::ifstream::in);
  if (!input) {
    std::cerr << "Error: input file could not be opened\n";
    return 1;
  }

  Points data = ReadPoints(input);
  input.close();

  char* output_file = argv[3];
  std::ofstream output;
  output.open(output_file, std::ifstream::out);
  if (!output) {
    std::cerr << "Error: output file could not be opened\n";
    return 1;
  }

  srand(123); // for reproducible results
  const size_t K = std::atoi(argv[1]);
  Points centroids = KMeans(data, K);

  WriteOutput(centroids, output);
  output.close();

  return 0;
}

// Gives random number in range [0..max_value]
size_t UniformRandom(const size_t max_value) {
  const size_t rnd = ((static_cast<size_t>(rand()) % 32768) << 17)
      | ((static_cast<size_t>(rand()) % 32768) << 2) | rand() % 4;
  return ((max_value + 1 == 0) ? rnd : rnd % (max_value + 1));
}

double Distance(const Point& point1, const Point& point2) {
  double distance_sqr = 0;
  size_t dimensions = point1.size();
  for (size_t i = 0; i < dimensions; ++i) {
    distance_sqr += (point1[i] - point2[i]) * (point1[i] - point2[i]);
  }

  return distance_sqr;
}

size_t FindNearestCentroid(const Points& centroids, const Point& point) {
  double min_distance = Distance(point, centroids[0]);
  size_t centroid_index = 0;
  for (size_t i = 1; i < centroids.size(); ++i) {
    double distance = Distance(point, centroids[i]);
    if (distance < min_distance) {
      min_distance = distance;
      centroid_index = i;
    }
  }
  return centroid_index;
}

// Calculates new centroid position as mean of positions of 3 random centroids
Point GetRandomPosition(const Points& centroids) {
  size_t K = centroids.size();
  int c1 = rand() % K;
  int c2 = rand() % K;
  int c3 = rand() % K;
  size_t dimensions = centroids[0].size();
  Point new_position(dimensions);
  for (size_t d = 0; d < dimensions; ++d) {
    new_position[d] = (centroids[c1][d] + centroids[c2][d] + centroids[c3][d]) / 3;
  }
  return new_position;
}

void FunctionForThreads(const size_t left_loop_bound,
                        const size_t right_loop_bound,
                        const size_t K,
                        const size_t dimensions,
                        const Points& centroids,
                        const Points& data,
                        Points& centroids_new,
                        std::vector<size_t>& clusters,
                        std::vector<size_t>& clusters_sizes,
                        bool& converged) {
  static std::mutex critical_section_in_the_end_mutex;
  bool converged_tmp = true;
  Points centroids_tmp(K, Point(dimensions));
  std::vector<size_t> clusters_sizes_tmp(K);

  for (size_t i = left_loop_bound; i < right_loop_bound; ++i) {
    size_t nearest_cluster = FindNearestCentroid(centroids, data[i]);
    if (clusters[i] != nearest_cluster) {
      clusters[i] = nearest_cluster;
      converged_tmp = false;
    }

    for (size_t d = 0; d < dimensions; ++d) {
      centroids_tmp[clusters[i]][d] += data[i][d];
    }
    ++clusters_sizes_tmp[clusters[i]];
  }

  {
    std::unique_lock<std::mutex> mutex_wrapper(critical_section_in_the_end_mutex);
    converged = converged_tmp && converged;
    for (size_t i = 0; i < K; ++i) {
      for (size_t d = 0; d < dimensions; ++d) {
        centroids_new[i][d] += centroids_tmp[i][d];
      }
      clusters_sizes[i] += clusters_sizes_tmp[i];
    }
  }
}

Points KMeans(const Points& data, const size_t K) {
  static constexpr size_t kThreadsNum = 2;
  const size_t data_size = data.size();
  const size_t dimensions = data[0].size();
  std::vector<size_t> clusters(data_size);

  // Initialize centroids randomly at data points
  Points centroids(K);
  for (size_t i = 0; i < K; ++i) {
    const size_t data_index = UniformRandom(data_size - 1);
    centroids[i] = data[data_index];
  }

  bool converged = false;
  while (!converged) {
    converged = true;

    std::vector<size_t> clusters_sizes(K);
    Points centroids_new(K, Point(dimensions));

    const size_t iterations_count = data_size;
    const double portion_size = static_cast<double>(iterations_count) / kThreadsNum;
    std::vector<std::thread> threads;
    threads.reserve(kThreadsNum);
    for (size_t thread_index = 0; thread_index != kThreadsNum; ++thread_index) {
      const size_t left_loop_bound = thread_index * portion_size;
      const size_t right_loop_bound = (thread_index + 1) * portion_size;
      threads.emplace_back(FunctionForThreads, left_loop_bound, right_loop_bound, K, dimensions,
                           std::ref(centroids), std::ref(data), std::ref(centroids_new),
                           std::ref(clusters), std::ref(clusters_sizes), std::ref(converged));
    }

    for (size_t thread_index = 0; thread_index != kThreadsNum; ++thread_index) {
      threads[thread_index].join();
    }

    if (converged) {
      break;
    }

    centroids_new.swap(centroids);

    for (size_t i = 0; i < K; ++i) {
      if (clusters_sizes[i] != 0) {
        for (size_t d = 0; d < dimensions; ++d) {
          centroids[i][d] /= clusters_sizes[i];
        }
      }
    }

    for (size_t i = 0; i < K; ++i) {
      if (clusters_sizes[i] == 0) {
        centroids[i] = GetRandomPosition(centroids);
      }
    }
  }

  return centroids;
}

