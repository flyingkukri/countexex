#include "genTrainData.h"
#include <catch2/catch_test_macros.hpp>
#include <armadillo>

TEST_CASE("repeatDataLabels") {
    SECTION("Test that dimension are correct") {
        std::vector<int> importance = {1, 2, 3};
        // Create a matrix with 3 rows and 4 columns
        arma::fmat data(3, 4, arma::fill::zeros);
        data.at(0, 0) = 0.;
        data.at(0, 1) = 1.;
        data.at(0, 2) = 1.;
        data.at(0, 3) = 2.;
        arma::Row<size_t> labels = ("0 1 0 1");
        MdpInfo mdpInfo;
        mdpInfo.imps = importance;
        std::pair<arma::fmat, arma::Row<size_t>> result = repeatDataLabels(data, labels, mdpInfo);
        //  we should have 1 + 2 * 2 + 3 = 8 rows
        REQUIRE(result.first.n_rows == 2);
        REQUIRE(result.first.n_cols == 8);
        REQUIRE(result.second.n_rows == 1);
        REQUIRE(result.second.n_cols == 8);
    }
}

TEST_CASE("createDataLabels") {
    SECTION("Test that labels get set correctly") {
        // Create a matrix with 3 rows (dimensions) and 10 columns (observations)
    //    arma_rng::set_seed_random(); 
        float arr[] = {
            1., 2., 4.,
            5., 6., 7.,
            8., 9., 10.,
            11., 12., 13.,
            3., 14., 15.,
            4., 53., 16.,
            17., 18., 19.,
            43., 20., 21.,
            100., 0.01, 22.,
            23., 24., 25.
        };
        arma::fmat allPairs(arr, 3, 10, false);
        arma::uvec indices = {1, 2, 5, 7, 9};
        arma::fmat strategyPairs = allPairs.cols(indices);
        arma::Row<size_t> labels = createDataLabels(allPairs, strategyPairs);
        for (size_t i = 0; i < labels.n_cols; ++i) {
            if (arma::any(indices == i)) {
                REQUIRE(labels(i) == 1);
            } else {
                REQUIRE(labels(i) == 0);
            }
        }
    }
}

//TEST_CASE("createMatrixeFromValueMapTest") {
//    SECTION("Main Test") {
//        std::map<std::string, std::variant<std::vector<int>, std::vector<bool>>> value_map;
//        value_map["imps"] = std::vector<int>{1, 2, 3};
//        value_map["action"] = std::vector<int>{3, 2, 1};
//        value_map["a"] = std::vector<int>{5, 2, 1};
//        value_map["b"] = std::vector<int>{1, 2, 3};
//        value_map["lol"] = std::vector<bool>{true, false, true};
//    }
//}