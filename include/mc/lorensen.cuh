#pragma once

#include "mc/base.cuh"

namespace mc {

class Lorensen : public MCBase {
  public:
    static constexpr size_t max_triangles = 4;
    static constexpr size_t max_len = max_triangles * 3;

    void run(const thrust::device_vector<uint8_t> &cases_dv,
             const thrust::device_vector<uint> &cell_idx_dv, float3 *v,
             const float *values, const float3 *points, const uint *cells,
             float level) override;

    size_t get_max_triangles() const override { return max_triangles; }

  private:
    static constexpr int edges[] = {0, 1, 1, 3, 2, 3, 0, 2, 4, 5, 5, 7,
                                    6, 7, 4, 6, 0, 4, 1, 5, 3, 7, 2, 6};

    static constexpr int edge_table[] = {
        0b000000000000, 0b000100001001, 0b001000000011, 0b001100001010,
        0b100000001100, 0b100100000101, 0b101000001111, 0b101100000110,
        0b010000000110, 0b010100001111, 0b011000000101, 0b011100001100,
        0b110000001010, 0b110100000011, 0b111000001001, 0b111100000000,
        0b000110010000, 0b000010011001, 0b001110010011, 0b001010011010,
        0b100110011100, 0b100010010101, 0b101110011111, 0b101010010110,
        0b010110010110, 0b010010011111, 0b011110010101, 0b011010011100,
        0b110110011010, 0b110010010011, 0b111110011001, 0b111010010000,
        0b001000110000, 0b001100111001, 0b000000110011, 0b000100111010,
        0b101000111100, 0b101100110101, 0b100000111111, 0b100100110110,
        0b011000110110, 0b011100111111, 0b010000110101, 0b010100111100,
        0b111000111010, 0b111100110011, 0b110000111001, 0b110100110000,
        0b001110100000, 0b001010101001, 0b000110100011, 0b000010101010,
        0b101110101100, 0b101010100101, 0b100110101111, 0b100010100110,
        0b011110100110, 0b011010101111, 0b010110100101, 0b010010101100,
        0b111110101010, 0b111010100011, 0b110110101001, 0b110010100000,
        0b100011000000, 0b100111001001, 0b101011000011, 0b101111001010,
        0b000011001100, 0b000111000101, 0b001011001111, 0b001111000110,
        0b110011000110, 0b110111001111, 0b111011000101, 0b111111001100,
        0b010011001010, 0b010111000011, 0b011011001001, 0b011111000000,
        0b100101010000, 0b100001011001, 0b101101010011, 0b101001011010,
        0b000101011100, 0b000001010101, 0b001101011111, 0b001001010110,
        0b110101010110, 0b110001011111, 0b111101010101, 0b111001011100,
        0b010101011010, 0b010001010011, 0b011101011001, 0b011001010000,
        0b101011110000, 0b101111111001, 0b100011110011, 0b100111111010,
        0b001011111100, 0b001111110101, 0b000011111111, 0b000111110110,
        0b111011110110, 0b111111111111, 0b110011110101, 0b110111111100,
        0b011011111010, 0b011111110011, 0b010011111001, 0b010111110000,
        0b101101100000, 0b101001101001, 0b100101100011, 0b100001101010,
        0b001101101100, 0b001001100101, 0b000101101111, 0b000001100110,
        0b111101100110, 0b111001101111, 0b110101100101, 0b110001101100,
        0b011101101010, 0b011001100011, 0b010101101001, 0b010001100000,
        0b010001100000, 0b010101101001, 0b011001100011, 0b011101101010,
        0b110001101100, 0b110101100101, 0b111001101111, 0b111101100110,
        0b000001100110, 0b000101101111, 0b001001100101, 0b001101101100,
        0b100001101010, 0b100101100011, 0b101001101001, 0b101101100000,
        0b010111110000, 0b010011111001, 0b011111110011, 0b011011111010,
        0b110111111100, 0b110011110101, 0b111111111111, 0b111011110110,
        0b000111110110, 0b000011111111, 0b001111110101, 0b001011111100,
        0b100111111010, 0b100011110011, 0b101111111001, 0b101011110000,
        0b011001010000, 0b011101011001, 0b010001010011, 0b010101011010,
        0b111001011100, 0b111101010101, 0b110001011111, 0b110101010110,
        0b001001010110, 0b001101011111, 0b000001010101, 0b000101011100,
        0b101001011010, 0b101101010011, 0b100001011001, 0b100101010000,
        0b011111000000, 0b011011001001, 0b010111000011, 0b010011001010,
        0b111111001100, 0b111011000101, 0b110111001111, 0b110011000110,
        0b001111000110, 0b001011001111, 0b000111000101, 0b000011001100,
        0b101111001010, 0b101011000011, 0b100111001001, 0b100011000000,
        0b110010100000, 0b110110101001, 0b111010100011, 0b111110101010,
        0b010010101100, 0b010110100101, 0b011010101111, 0b011110100110,
        0b100010100110, 0b100110101111, 0b101010100101, 0b101110101100,
        0b000010101010, 0b000110100011, 0b001010101001, 0b001110100000,
        0b110100110000, 0b110000111001, 0b111100110011, 0b111000111010,
        0b010100111100, 0b010000110101, 0b011100111111, 0b011000110110,
        0b100100110110, 0b100000111111, 0b101100110101, 0b101000111100,
        0b000100111010, 0b000000110011, 0b001100111001, 0b001000110000,
        0b111010010000, 0b111110011001, 0b110010010011, 0b110110011010,
        0b011010011100, 0b011110010101, 0b010010011111, 0b010110010110,
        0b101010010110, 0b101110011111, 0b100010010101, 0b100110011100,
        0b001010011010, 0b001110010011, 0b000010011001, 0b000110010000,
        0b111100000000, 0b111000001001, 0b110100000011, 0b110000001010,
        0b011100001100, 0b011000000101, 0b010100001111, 0b010000000110,
        0b101100000110, 0b101000001111, 0b100100000101, 0b100000001100,
        0b001100001010, 0b001000000011, 0b000100001001, 0b000000000000,
    };

    static constexpr int tri_table[] = {
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  8,  3,  -1, -1, -1,
        -1, -1, -1, -1, -1, -1, 1,  9,  0,  -1, -1, -1, -1, -1, -1, -1, -1, -1,
        9,  3,  1,  9,  8,  3,  -1, -1, -1, -1, -1, -1, 3,  11, 2,  -1, -1, -1,
        -1, -1, -1, -1, -1, -1, 2,  8,  11, 2,  0,  8,  -1, -1, -1, -1, -1, -1,
        2,  3,  11, 0,  1,  9,  -1, -1, -1, -1, -1, -1, 2,  1,  9,  2,  9,  11,
        11, 9,  8,  -1, -1, -1, 2,  10, 1,  -1, -1, -1, -1, -1, -1, -1, -1, -1,
        3,  0,  8,  1,  2,  10, -1, -1, -1, -1, -1, -1, 0,  10, 9,  0,  2,  10,
        -1, -1, -1, -1, -1, -1, 3,  2,  10, 3,  10, 8,  8,  10, 9,  -1, -1, -1,
        11, 1,  3,  11, 10, 1,  -1, -1, -1, -1, -1, -1, 1,  0,  8,  1,  8,  10,
        10, 8,  11, -1, -1, -1, 0,  3,  11, 0,  11, 9,  9,  11, 10, -1, -1, -1,
        9,  8,  11, 9,  11, 10, -1, -1, -1, -1, -1, -1, 8,  4,  7,  -1, -1, -1,
        -1, -1, -1, -1, -1, -1, 0,  7,  3,  0,  4,  7,  -1, -1, -1, -1, -1, -1,
        9,  0,  1,  8,  4,  7,  -1, -1, -1, -1, -1, -1, 9,  4,  7,  9,  7,  1,
        1,  7,  3,  -1, -1, -1, 3,  11, 2,  7,  8,  4,  -1, -1, -1, -1, -1, -1,
        7,  11, 2,  7,  2,  4,  4,  2,  0,  -1, -1, -1, 0,  1,  9,  4,  7,  8,
        3,  11, 2,  -1, -1, -1, 1,  11, 2,  1,  9,  11, 7,  11, 9,  7,  9,  4,
        2,  10, 1,  7,  8,  4,  -1, -1, -1, -1, -1, -1, 7,  0,  4,  7,  3,  0,
        2,  10, 1,  -1, -1, -1, 10, 0,  2,  10, 9,  0,  4,  7,  8,  -1, -1, -1,
        4,  7,  3,  2,  10, 3,  10, 4,  3,  4,  10, 9,  1,  11, 10, 1,  3,  11,
        8,  4,  7,  -1, -1, -1, 4,  7,  0,  0,  7,  10, 10, 7,  11, 0,  10, 1,
        0,  3,  8,  9,  4,  7,  11, 9,  7,  10, 9,  11, 9,  4,  7,  11, 9,  7,
        10, 9,  11, -1, -1, -1, 4,  9,  5,  -1, -1, -1, -1, -1, -1, -1, -1, -1,
        0,  8,  3,  4,  9,  5,  -1, -1, -1, -1, -1, -1, 1,  4,  0,  1,  5,  4,
        -1, -1, -1, -1, -1, -1, 4,  8,  3,  4,  3,  5,  5,  3,  1,  -1, -1, -1,
        3,  11, 2,  4,  9,  5,  -1, -1, -1, -1, -1, -1, 2,  8,  11, 2,  0,  8,
        9,  5,  4,  -1, -1, -1, 4,  1,  5,  4,  0,  1,  3,  11, 2,  -1, -1, -1,
        11, 4,  8,  1,  4,  11, 5,  4,  1,  2,  1,  11, 9,  5,  4,  10, 1,  2,
        -1, -1, -1, -1, -1, -1, 9,  5,  4,  8,  3,  0,  1,  2,  10, -1, -1, -1,
        10, 5,  4,  10, 4,  2,  2,  4,  0,  -1, -1, -1, 2,  8,  3,  2,  10, 8,
        4,  8,  10, 4,  10, 5,  11, 1,  3,  11, 10, 1,  5,  4,  9,  -1, -1, -1,
        9,  5,  4,  1,  0,  8,  1,  8,  10, 10, 8,  11, 3,  11, 10, 5,  4,  10,
        4,  3,  10, 3,  4,  0,  10, 5,  4,  8,  10, 4,  11, 10, 8,  -1, -1, -1,
        8,  5,  7,  8,  9,  5,  -1, -1, -1, -1, -1, -1, 0,  9,  5,  0,  5,  3,
        3,  5,  7,  -1, -1, -1, 8,  0,  1,  8,  1,  7,  7,  1,  5,  -1, -1, -1,
        5,  7,  3,  1,  5,  3,  -1, -1, -1, -1, -1, -1, 5,  8,  9,  5,  7,  8,
        11, 2,  3,  -1, -1, -1, 11, 2,  0,  9,  5,  0,  5,  11, 0,  11, 5,  7,
        8,  0,  3,  7,  11, 2,  1,  7,  2,  5,  7,  1,  7,  11, 2,  1,  7,  2,
        5,  7,  1,  -1, -1, -1, 8,  5,  7,  8,  9,  5,  1,  2,  10, -1, -1, -1,
        1,  2,  10, 0,  9,  5,  0,  5,  3,  3,  5,  7,  2,  8,  0,  5,  8,  2,
        7,  8,  5,  10, 5,  2,  3,  2,  10, 5,  3,  10, 7,  3,  5,  -1, -1, -1,
        3,  9,  1,  8,  9,  3,  11, 10, 5,  7,  11, 5,  11, 10, 7,  10, 5,  7,
        1,  0,  9,  -1, -1, -1, 5,  7,  10, 7,  11, 10, 8,  0,  3,  -1, -1, -1,
        5,  7,  10, 7,  11, 10, -1, -1, -1, -1, -1, -1, 6,  11, 7,  -1, -1, -1,
        -1, -1, -1, -1, -1, -1, 8,  3,  0,  11, 7,  6,  -1, -1, -1, -1, -1, -1,
        1,  9,  0,  6,  11, 7,  -1, -1, -1, -1, -1, -1, 9,  3,  1,  9,  8,  3,
        7,  6,  11, -1, -1, -1, 3,  6,  2,  3,  7,  6,  -1, -1, -1, -1, -1, -1,
        8,  7,  6,  8,  6,  0,  0,  6,  2,  -1, -1, -1, 6,  3,  7,  6,  2,  3,
        1,  9,  0,  -1, -1, -1, 9,  8,  7,  9,  2,  1,  9,  7,  2,  6,  2,  7,
        2,  10, 1,  6,  11, 7,  -1, -1, -1, -1, -1, -1, 2,  10, 1,  0,  8,  3,
        11, 7,  6,  -1, -1, -1, 0,  10, 9,  0,  2,  10, 11, 7,  6,  -1, -1, -1,
        11, 7,  6,  3,  2,  10, 3,  10, 8,  8,  10, 9,  6,  10, 1,  6,  1,  7,
        7,  1,  3,  -1, -1, -1, 0,  10, 1,  0,  8,  10, 6,  10, 8,  6,  8,  7,
        9,  0,  10, 10, 0,  7,  7,  0,  3,  10, 7,  6,  8,  7,  6,  10, 8,  6,
        9,  8,  10, -1, -1, -1, 4,  11, 8,  4,  6,  11, -1, -1, -1, -1, -1, -1,
        11, 3,  0,  11, 0,  6,  6,  0,  4,  -1, -1, -1, 11, 4,  6,  11, 8,  4,
        0,  1,  9,  -1, -1, -1, 1,  11, 3,  4,  11, 1,  6,  11, 4,  9,  4,  1,
        3,  8,  4,  3,  4,  2,  2,  4,  6,  -1, -1, -1, 0,  4,  6,  0,  6,  2,
        -1, -1, -1, -1, -1, -1, 0,  1,  9,  3,  8,  4,  3,  4,  2,  2,  4,  6,
        2,  1,  9,  4,  2,  9,  6,  2,  4,  -1, -1, -1, 4,  11, 8,  4,  6,  11,
        10, 1,  2,  -1, -1, -1, 11, 3,  2,  6,  10, 1,  0,  6,  1,  4,  6,  0,
        2,  8,  0,  11, 8,  2,  10, 9,  4,  6,  10, 4,  4,  6,  9,  6,  10, 9,
        11, 3,  2,  -1, -1, -1, 1,  3,  8,  1,  6,  10, 1,  8,  6,  4,  6,  8,
        6,  10, 1,  0,  6,  1,  4,  6,  0,  -1, -1, -1, 10, 9,  6,  9,  4,  6,
        0,  3,  8,  -1, -1, -1, 10, 9,  6,  9,  4,  6,  -1, -1, -1, -1, -1, -1,
        7,  6,  11, 5,  4,  9,  -1, -1, -1, -1, -1, -1, 7,  6,  11, 3,  0,  8,
        4,  9,  5,  -1, -1, -1, 1,  4,  0,  1,  5,  4,  6,  11, 7,  -1, -1, -1,
        7,  6,  11, 4,  8,  3,  4,  3,  5,  5,  3,  1,  3,  6,  2,  3,  7,  6,
        4,  9,  5,  -1, -1, -1, 8,  7,  4,  0,  9,  5,  6,  0,  5,  2,  0,  6,
        5,  4,  1,  1,  4,  0,  7,  6,  2,  7,  2,  3,  1,  5,  2,  5,  6,  2,
        4,  8,  7,  -1, -1, -1, 5,  4,  9,  1,  2,  10, 6,  11, 7,  -1, -1, -1,
        0,  8,  3,  1,  2,  10, 4,  9,  5,  6,  11, 7,  10, 5,  6,  2,  11, 7,
        4,  2,  7,  0,  2,  4,  10, 5,  6,  11, 3,  2,  4,  8,  7,  -1, -1, -1,
        5,  4,  9,  6,  10, 1,  6,  1,  7,  7,  1,  3,  8,  7,  4,  9,  1,  0,
        6,  10, 5,  -1, -1, -1, 3,  7,  0,  7,  4,  0,  6,  10, 5,  -1, -1, -1,
        8,  7,  4,  10, 5,  6,  -1, -1, -1, -1, -1, -1, 5,  6,  11, 5,  11, 9,
        9,  11, 8,  -1, -1, -1, 0,  9,  3,  9,  11, 3,  11, 9,  5,  6,  11, 5,
        0,  1,  5,  6,  11, 5,  11, 0,  5,  0,  11, 8,  5,  6,  11, 3,  5,  11,
        1,  5,  3,  -1, -1, -1, 2,  3,  6,  6,  3,  9,  9,  3,  8,  6,  9,  5,
        0,  9,  5,  6,  0,  5,  2,  0,  6,  -1, -1, -1, 6,  2,  5,  2,  1,  5,
        3,  8,  0,  -1, -1, -1, 1,  5,  2,  5,  6,  2,  -1, -1, -1, -1, -1, -1,
        10, 1,  2,  5,  6,  11, 5,  11, 9,  9,  11, 8,  11, 3,  2,  10, 5,  6,
        0,  9,  1,  -1, -1, -1, 0,  2,  8,  2,  11, 8,  10, 5,  6,  -1, -1, -1,
        5,  6,  10, 3,  2,  11, -1, -1, -1, -1, -1, -1, 8,  9,  3,  9,  1,  3,
        5,  6,  10, -1, -1, -1, 0,  9,  1,  6,  10, 5,  -1, -1, -1, -1, -1, -1,
        3,  8,  0,  6,  10, 5,  -1, -1, -1, -1, -1, -1, 5,  6,  10, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, 10, 6,  5,  -1, -1, -1, -1, -1, -1, -1, -1, -1,
        0,  8,  3,  5,  10, 6,  -1, -1, -1, -1, -1, -1, 1,  9,  0,  5,  10, 6,
        -1, -1, -1, -1, -1, -1, 3,  9,  8,  3,  1,  9,  10, 6,  5,  -1, -1, -1,
        10, 6,  5,  11, 2,  3,  -1, -1, -1, -1, -1, -1, 8,  2,  0,  8,  11, 2,
        6,  5,  10, -1, -1, -1, 2,  3,  11, 6,  5,  10, 1,  9,  0,  -1, -1, -1,
        2,  1,  10, 11, 6,  5,  9,  11, 5,  8,  11, 9,  2,  5,  1,  2,  6,  5,
        -1, -1, -1, -1, -1, -1, 5,  2,  6,  5,  1,  2,  0,  8,  3,  -1, -1, -1,
        5,  9,  0,  5,  0,  6,  6,  0,  2,  -1, -1, -1, 6,  3,  2,  9,  3,  6,
        8,  3,  9,  5,  9,  6,  11, 6,  5,  11, 5,  3,  3,  5,  1,  -1, -1, -1,
        5,  1,  0,  5,  11, 6,  5,  0,  11, 8,  11, 0,  3,  9,  0,  3,  11, 9,
        5,  9,  11, 5,  11, 6,  11, 6,  5,  9,  11, 5,  8,  11, 9,  -1, -1, -1,
        4,  7,  8,  6,  5,  10, -1, -1, -1, -1, -1, -1, 0,  7,  3,  0,  4,  7,
        5,  10, 6,  -1, -1, -1, 4,  7,  8,  0,  1,  9,  5,  10, 6,  -1, -1, -1,
        9,  4,  5,  1,  10, 6,  7,  1,  6,  3,  1,  7,  6,  5,  10, 2,  3,  11,
        7,  8,  4,  -1, -1, -1, 6,  5,  10, 7,  11, 2,  7,  2,  4,  4,  2,  0,
        3,  8,  0,  10, 2,  1,  5,  9,  4,  7,  11, 6,  9,  4,  5,  10, 2,  1,
        7,  11, 6,  -1, -1, -1, 2,  5,  1,  2,  6,  5,  7,  8,  4,  -1, -1, -1,
        1,  4,  5,  0,  4,  1,  2,  6,  7,  3,  2,  7,  4,  7,  8,  5,  9,  0,
        5,  0,  6,  6,  0,  2,  2,  6,  3,  6,  7,  3,  5,  9,  4,  -1, -1, -1,
        11, 6,  7,  3,  8,  4,  5,  3,  4,  1,  3,  5,  0,  4,  1,  4,  5,  1,
        7,  11, 6,  -1, -1, -1, 11, 6,  7,  8,  0,  3,  5,  9,  4,  -1, -1, -1,
        11, 6,  7,  9,  4,  5,  -1, -1, -1, -1, -1, -1, 6,  9,  10, 6,  4,  9,
        -1, -1, -1, -1, -1, -1, 6,  9,  10, 6,  4,  9,  8,  3,  0,  -1, -1, -1,
        1,  10, 6,  1,  6,  0,  0,  6,  4,  -1, -1, -1, 8,  3,  1,  10, 6,  1,
        6,  8,  1,  8,  6,  4,  9,  6,  4,  9,  10, 6,  2,  3,  11, -1, -1, -1,
        0,  8,  2,  2,  8,  11, 4,  9,  10, 4,  10, 6,  2,  3,  11, 1,  10, 6,
        1,  6,  0,  0,  6,  4,  8,  11, 4,  11, 6,  4,  2,  1,  10, -1, -1, -1,
        9,  1,  2,  9,  2,  4,  4,  2,  6,  -1, -1, -1, 9,  1,  0,  4,  8,  3,
        2,  4,  3,  6,  4,  2,  6,  4,  0,  2,  6,  0,  -1, -1, -1, -1, -1, -1,
        4,  8,  3,  2,  4,  3,  6,  4,  2,  -1, -1, -1, 3,  11, 1,  1,  11, 4,
        4,  11, 6,  1,  4,  9,  6,  4,  11, 4,  8,  11, 9,  1,  0,  -1, -1, -1,
        0,  3,  11, 6,  0,  11, 4,  0,  6,  -1, -1, -1, 8,  11, 4,  11, 6,  4,
        -1, -1, -1, -1, -1, -1, 6,  7,  8,  6,  8,  10, 10, 8,  9,  -1, -1, -1,
        10, 0,  9,  7,  0,  10, 3,  0,  7,  6,  7,  10, 1,  10, 0,  10, 8,  0,
        8,  10, 6,  7,  8,  6,  1,  10, 6,  7,  1,  6,  3,  1,  7,  -1, -1, -1,
        6,  7,  11, 10, 2,  3,  8,  10, 3,  9,  10, 8,  9,  10, 0,  10, 2,  0,
        6,  7,  11, -1, -1, -1, 1,  10, 2,  3,  8,  0,  6,  7,  11, -1, -1, -1,
        1,  10, 2,  7,  11, 6,  -1, -1, -1, -1, -1, -1, 7,  8,  9,  1,  2,  9,
        2,  7,  9,  7,  2,  6,  7,  3,  6,  3,  2,  6,  0,  9,  1,  -1, -1, -1,
        6,  7,  8,  0,  6,  8,  2,  6,  0,  -1, -1, -1, 2,  6,  3,  6,  7,  3,
        -1, -1, -1, -1, -1, -1, 1,  3,  9,  3,  8,  9,  11, 6,  7,  -1, -1, -1,
        0,  9,  1,  7,  11, 6,  -1, -1, -1, -1, -1, -1, 0,  3,  8,  6,  7,  11,
        -1, -1, -1, -1, -1, -1, 7,  11, 6,  -1, -1, -1, -1, -1, -1, -1, -1, -1,
        10, 7,  5,  10, 11, 7,  -1, -1, -1, -1, -1, -1, 10, 7,  5,  10, 11, 7,
        3,  0,  8,  -1, -1, -1, 7,  10, 11, 7,  5,  10, 9,  0,  1,  -1, -1, -1,
        1,  9,  3,  3,  9,  8,  5,  10, 11, 5,  11, 7,  10, 2,  3,  10, 3,  5,
        5,  3,  7,  -1, -1, -1, 0,  8,  2,  2,  8,  5,  5,  8,  7,  2,  5,  10,
        10, 2,  1,  5,  9,  0,  3,  5,  0,  7,  5,  3,  7,  5,  8,  5,  9,  8,
        10, 2,  1,  -1, -1, -1, 2,  11, 7,  2,  7,  1,  1,  7,  5,  -1, -1, -1,
        3,  0,  8,  2,  11, 7,  2,  7,  1,  1,  7,  5,  0,  2,  11, 0,  5,  9,
        0,  11, 5,  7,  5,  11, 9,  8,  5,  8,  7,  5,  3,  2,  11, -1, -1, -1,
        3,  7,  5,  3,  5,  1,  -1, -1, -1, -1, -1, -1, 1,  0,  8,  7,  1,  8,
        5,  1,  7,  -1, -1, -1, 5,  9,  0,  3,  5,  0,  7,  5,  3,  -1, -1, -1,
        7,  5,  8,  5,  9,  8,  -1, -1, -1, -1, -1, -1, 4,  5,  10, 4,  10, 8,
        8,  10, 11, -1, -1, -1, 10, 11, 3,  10, 4,  5,  10, 3,  4,  0,  4,  3,
        4,  5,  9,  8,  0,  1,  10, 8,  1,  11, 8,  10, 3,  1,  11, 1,  10, 11,
        9,  4,  5,  -1, -1, -1, 3,  8,  2,  8,  10, 2,  10, 8,  4,  5,  10, 4,
        4,  5,  10, 2,  4,  10, 0,  4,  2,  -1, -1, -1, 4,  5,  9,  0,  3,  8,
        10, 2,  1,  -1, -1, -1, 4,  5,  9,  2,  1,  10, -1, -1, -1, -1, -1, -1,
        8,  4,  11, 11, 4,  1,  1,  4,  5,  11, 1,  2,  5,  1,  4,  1,  0,  4,
        2,  11, 3,  -1, -1, -1, 11, 8,  2,  8,  0,  2,  4,  5,  9,  -1, -1, -1,
        2,  11, 3,  5,  9,  4,  -1, -1, -1, -1, -1, -1, 3,  8,  4,  5,  3,  4,
        1,  3,  5,  -1, -1, -1, 0,  4,  1,  4,  5,  1,  -1, -1, -1, -1, -1, -1,
        3,  8,  0,  5,  9,  4,  -1, -1, -1, -1, -1, -1, 5,  9,  4,  -1, -1, -1,
        -1, -1, -1, -1, -1, -1, 7,  4,  9,  7,  9,  11, 11, 9,  10, -1, -1, -1,
        8,  3,  0,  7,  4,  9,  7,  9,  11, 11, 9,  10, 0,  7,  4,  10, 7,  0,
        11, 7,  10, 1,  10, 0,  10, 11, 1,  11, 3,  1,  7,  4,  8,  -1, -1, -1,
        3,  7,  4,  3,  10, 2,  3,  4,  10, 9,  10, 4,  2,  0,  10, 0,  9,  10,
        8,  7,  4,  -1, -1, -1, 4,  0,  7,  0,  3,  7,  1,  10, 2,  -1, -1, -1,
        1,  10, 2,  4,  8,  7,  -1, -1, -1, -1, -1, -1, 2,  11, 1,  11, 9,  1,
        9,  11, 7,  4,  9,  7,  9,  1,  0,  8,  7,  4,  2,  11, 3,  -1, -1, -1,
        2,  11, 7,  4,  2,  7,  0,  2,  4,  -1, -1, -1, 2,  11, 3,  4,  8,  7,
        -1, -1, -1, -1, -1, -1, 7,  4,  9,  1,  7,  9,  3,  7,  1,  -1, -1, -1,
        1,  0,  9,  7,  4,  8,  -1, -1, -1, -1, -1, -1, 3,  7,  0,  7,  4,  0,
        -1, -1, -1, -1, -1, -1, 7,  4,  8,  -1, -1, -1, -1, -1, -1, -1, -1, -1,
        11, 8,  9,  10, 11, 9,  -1, -1, -1, -1, -1, -1, 11, 3,  0,  9,  11, 0,
        10, 11, 9,  -1, -1, -1, 8,  0,  1,  10, 8,  1,  11, 8,  10, -1, -1, -1,
        3,  1,  11, 1,  10, 11, -1, -1, -1, -1, -1, -1, 10, 2,  3,  8,  10, 3,
        9,  10, 8,  -1, -1, -1, 9,  10, 0,  10, 2,  0,  -1, -1, -1, -1, -1, -1,
        8,  0,  3,  10, 2,  1,  -1, -1, -1, -1, -1, -1, 1,  10, 2,  -1, -1, -1,
        -1, -1, -1, -1, -1, -1, 9,  1,  2,  11, 9,  2,  8,  9,  11, -1, -1, -1,
        11, 3,  2,  9,  1,  0,  -1, -1, -1, -1, -1, -1, 11, 8,  2,  8,  0,  2,
        -1, -1, -1, -1, -1, -1, 2,  11, 3,  -1, -1, -1, -1, -1, -1, -1, -1, -1,
        1,  3,  9,  3,  8,  9,  -1, -1, -1, -1, -1, -1, 0,  9,  1,  -1, -1, -1,
        -1, -1, -1, -1, -1, -1, 3,  8,  0,  -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    };

    static constexpr size_t edges_size = sizeof(edges) / sizeof(edges[0]);
    static constexpr size_t edge_table_size =
        sizeof(edge_table) / sizeof(edge_table[0]);
    static constexpr size_t tri_table_size =
        sizeof(tri_table) / sizeof(tri_table[0]);
};

}   // namespace mc