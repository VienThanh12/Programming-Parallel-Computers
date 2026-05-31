#pragma once

#include "ppc.h"

struct input {
    int ny;
    int nx;
    int hy;
    int hx;
    std::vector<float> input;
};

static input generate_raw_input(std::ifstream &input_file) {
    std::string header;
    CHECK_READ(getline(input_file, header));
    std::stringstream header_reader(header);
    int ny, nx, hy, hx;
    CHECK_READ(header_reader >> ny >> nx >> hy >> hx);
    CHECK_END(header_reader);

    std::vector<float> data(ny * nx);
    for (int y = 0; y < ny; y++) {
        std::string line;
        CHECK_READ(getline(input_file, line));
        std::stringstream line_reader(line);
        for (int x = 0; x < nx; x++) {
            float value;
            CHECK_READ(line_reader >> value);
            data[y * nx + x] = value;
        }
        CHECK_END(line_reader);
    }
    CHECK_END(input_file);

    return {ny, nx, hy, hx, data};
}

static input generate_random_input(std::ifstream &input_file) {
    ppc::random rng;
    int ny, nx, hy, hx;
    CHECK_READ(input_file >> ny >> nx >> hy >> hx);
    CHECK_END(input_file);

    std::vector<float> data(ny * nx);
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            data[y * nx + x] = rng.get_float();
        }
    }

    return {ny, nx, hy, hx, data};
}