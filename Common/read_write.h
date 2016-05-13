/*
 * read_write.h
 *
 *  Created on: 12 ��� 2016 �.
 *      Author: user
 */

#ifndef COMMON_READ_WRITE_H_
#define COMMON_READ_WRITE_H_

#include <fstream>
#include "types.h"

Points ReadPoints(std::ifstream& input);
void WriteOutput(const Points& centroids, std::ofstream& output);

#endif /* COMMON_READ_WRITE_H_ */
