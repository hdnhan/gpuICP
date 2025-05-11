#pragma once
#include <tuple>
#include <vector>

std::tuple<std::vector<float>, std::vector<float>>
computeRt(std::vector<float> const &H, float sx, float sy, float sz, float tx, float ty, float tz);