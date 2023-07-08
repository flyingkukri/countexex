#include "genTrainData.h"
#include <catch2/catch_test_macros.hpp>
#include <armadillo>

TEST_CASE("repeatDataLabels") {
    SECTION("Dimension are correct") {
        std::vector<int> importance = {1, 2, 3};
        // Create a matrix with 3 rows and 4 columns
        arma::mat data(3, 4, arma::fill::zeros);
        data[0, 0] = 0;
        data[1, 0] = 1;
        data[2, 0] = 1;
        data[3, 0] = 2;
        arma::Row<size_t> labels = ("0 1 0 1");
        
        std::pair<arma::mat, arma::Row<size_t>> result = repeatDataLabels(data, labels, importance);
        //  we should have 1 + 2 * 2 + 3 = 8 rows
        REQUIRE(result.first.n_rows == 8);
        REQUIRE(result.second.n_rows == 8);
    }
}