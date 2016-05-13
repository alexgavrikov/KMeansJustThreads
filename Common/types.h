/*
 * types.h
 *
 *  Created on: 12 ��� 2016 �.
 *      Author: user
 */

#ifndef COMMON_TYPES_H_
#define COMMON_TYPES_H_

#include <vector>

using Point = std::vector<double>;
using Points = std::vector<Point>;
using PointIterator = Points::iterator;
using PointConstIterator = Points::const_iterator;

#endif /* COMMON_TYPES_H_ */
