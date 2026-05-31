#include "pngio.h"
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "mf.h"
#include "tests.h"

ppc::Image8 handle_test_case(const char *testfile) {
    std::ifstream input_file(testfile);
    if (!input_file) {
        std::cerr << "Failed to open input file" << std::endl;
        std::exit(1);
    }

    std::string input_type;
    CHECK_READ(input_file >> input_type);
    if (input_type == "timeout") {
        input_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        CHECK_READ(input_file >> input_type);
    }

    input input;
    if (input_type == "raw") {
        input = generate_raw_input(input_file);
    } else if (input_type == "random") {
        input = generate_random_input(input_file);
    } else {
        std::cerr << "Invalid input type" << std::endl;
        std::exit(3);
    }

    std::vector<float> output(input.nx * input.ny);
    mf(input.ny, input.nx, input.hy, input.hx, input.input.data(), output.data());

    int ny = input.ny;
    int nx = input.nx;

    ppc::Image8 result;
    result.resize(ny, 2 * nx + 2, 3);

    // draw separating lines
    for (int y = 0; y < ny; ++y) {
        result.setlin3(y, nx, 1.0, 0.5, 0.0);
        result.setlin3(y, nx + 1, 1.0, 0.5, 0.0);
    }

    // draw the images
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            float c = input.input[x + nx * y];
            result.setlin3(y, x, c, c, c);
            c = output[y * nx + x];
            result.setlin3(y, nx + 2 + x, c, c, c);
        }
    }

    return result;
}

ppc::Image8 handle_image(const char *testfile) {
    ppc::Image8 im_input;
    ppc::read_image(im_input, testfile);

    if (im_input.nc != 3) {
        std::cerr << "Image " << testfile << " should have 3 color components." << std::endl;
        std::exit(1);
    }

    input instance;
    instance.nx = im_input.nx;
    instance.ny = im_input.ny;
    instance.hx = 5;
    instance.hy = 5;
    instance.input.resize(instance.nx * instance.ny);

    int ny = instance.ny;
    int nx = instance.nx;

    ppc::Image8 result;
    result.resize(ny, 2 * nx + 2, 3);

    // draw separating lines
    for (int y = 0; y < ny; ++y) {
        result.setlin3(y, nx, 1.0, 0.5, 0.0);
        result.setlin3(y, nx + 1, 1.0, 0.5, 0.0);
    }

    for (int channel = 0; channel < 3; ++channel) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                instance.input[x + y * nx] = im_input.get(y, x, channel);
            }
        }

        std::vector<float> output(ny * nx);
        mf(instance.ny, instance.nx, instance.hy, instance.hx, instance.input.data(), output.data());

        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                float c_img = instance.input[x + nx * y];
                float c_med = output[x + nx * y];
                result.set(y, x, channel, c_img);
                result.set(y, nx + 2 + x, channel, c_med);
            }
        }
    }

    return result;
}

void help(const char *prog) {
    std::cout << "usage: " << prog << " test.txt [output]" << std::endl;
    std::cout << "       " << prog << " test.png [output]" << std::endl;
    std::cout << "  If test is a png image, run the correlation on that image." << std::endl;
    std::cout << "  If test is a test case, run the correlation on that test case." << std::endl;
    std::exit(1);
}

int main(int argc, const char **argv) {
    if (argc != 2 && argc != 3) {
        help(argv[0]);
    }
    const char *testfile = argv[1];
    std::size_t len = strlen(testfile);
    if (len < 4) {
        std::cerr << "Input file name too short. Please use either a '*.txt' or a '*.png' file." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const char *imfile_output = "out.png";
    if (argc == 3) {
        imfile_output = argv[2];
    }

    auto result = [&]() {
        if (strcmp(testfile + len - 4, ".png") == 0) {
            return handle_image(testfile);
        } else if (strcmp(testfile + len - 4, ".txt") == 0) {
            return handle_test_case(testfile);
        } else {
            help(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }();

    write_image(result, imfile_output);

    std::cout << "Generated result '" << imfile_output << "'\n";

    return EXIT_SUCCESS;
}
