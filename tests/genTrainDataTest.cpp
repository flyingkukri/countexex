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

TEST_CASE("createMatrixFromValueMapTest") {
    SECTION("Main Test") {
        std::map<std::string, std::variant<std::vector<int>, std::vector<bool>>> value_map;
        value_map["imps"] = std::vector<int>{1, 2, 3};
        value_map["action"] = std::vector<int>{0, 1, 2};
        value_map["a"] = std::vector<int>{5, 2, 1};
        value_map["b"] = std::vector<int>{1, 2, 3};
        value_map["lol"] = std::vector<bool>{true, false, true};

        MdpInfo mdpInfo;
        mdpInfo.imps = std::vector<int>{1, 1, 1};
        mdpInfo.numOfActId = 3;
        arma::mat result = createMatrixFromValueMap(value_map, mdpInfo);
        
        arma::mat test = 
        {
            {1, 2, 3},
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, 1},
            {5, 2, 1},
            {1, 2, 3},
            {1, 0, 1}
        };

        REQUIRE(result.n_rows == test.n_rows);
        REQUIRE(result.n_cols == test.n_cols);
    
        REQUIRE(arma::approx_equal(result, test, "absdiff", 0.0001));
    }
}

TEST_CASE("categoricalFeatureOneHotEncodingTest") {
    SECTION("Well ordered") {
        arma::mat baseMat;
        MdpInfo mdpInfo;
        mdpInfo.numOfActId = 5;

        std::vector<int> actionVec1 = {0, 1, 2, 3, 4};
        arma::mat test1 = 
        {
            {1, 2, 3, 4, 5},
            {1, 0, 0, 0, 0},
            {0, 1, 0, 0, 0},
            {0, 0, 1, 0, 0},
            {0, 0, 0, 1, 0},
            {0, 0, 0, 0, 1}
        };

        std::vector<int> actionVec2 = {1, 3, 2, 4, 0};
        arma::mat test2 = 
        {
            {1, 2, 3, 4, 5},
            {0, 1, 0, 0, 0},
            {0, 0, 0, 1, 0},
            {0, 0, 1, 0, 0},
            {0, 0, 0, 0, 1},
            {1, 0, 0, 0, 0}
        };

        struct testCase {
            std::vector<int> actionVec;
            arma::mat test;
        };

        struct testCase testCases[] = {
            {actionVec1, test1}//,
            //{actionVec2, test2}
        };

        int numTestCases = sizeof(testCases) / sizeof(testCases[0]); 

        for (int i = 0; i < numTestCases; i++) {
            baseMat = 
            {
                {1, 2, 3, 4, 5}
            }; 
            std::vector<int> actionVec_ = testCases[i].actionVec;
            std::variant<std::vector<int>, std::vector<bool>> actionVec;
            actionVec = actionVec_;
            arma::mat test = testCases[i].test;
            
            categoricalFeatureOneHotEncoding(baseMat, mdpInfo, actionVec);
            CAPTURE(actionVec);
            CAPTURE(i);
            assert(baseMat.n_rows == test.n_rows);
            assert(baseMat.n_cols == test.n_cols);
            assert(arma::approx_equal(baseMat, test, "absdiff", 0.0001));
        }
    }
}

TEST_CASE("createTrainingDataTest") {
    SECTION("Main Test") {
        std::map<std::string, std::variant<std::vector<int>, std::vector<bool>>> value_map;
        value_map["imps"] = std::vector<int>{0, 1, 2};
        value_map["action"] = std::vector<int>{0, 1, 2};
        value_map["a"] = std::vector<int>{5, 2, 1};
        value_map["b"] = std::vector<int>{1, 2, 3};
        value_map["lol"] = std::vector<bool>{true, false, true};

        std::map<std::string, std::variant<std::vector<int>, std::vector<bool>>> value_map_submdp;
        value_map_submdp["imps"] = std::vector<int>{0, 2};
        value_map_submdp["action"] = std::vector<int>{0, 2};
        value_map_submdp["a"] = std::vector<int>{5, 1};
        value_map_submdp["b"] = std::vector<int>{1, 3};
        value_map_submdp["lol"] = std::vector<bool>{true, true};

        MdpInfo mdpInfo;
        mdpInfo.imps = std::vector<int>{1, 2, 3};
        mdpInfo.numOfActId = 3;
        auto result = createTrainingData(value_map, value_map_submdp, mdpInfo);
        arma::mat data = result.first;
        arma::Row<size_t> labels = result.second;
        
        arma::mat test = 
        {
//            {1, 2, 2, 3, 3, 3},
            {1, 0, 0, 0, 0, 0},
            {0, 1, 1, 0, 0, 0},
            {0, 0, 0, 1, 1, 1},
            {5, 2, 2, 1, 1, 1},
            {1, 2, 2, 3, 3, 3},
            {1, 0, 0, 1, 1, 1}
        };
        

        arma::Row<size_t> testLabels = {1, 0, 0, 1, 1, 1};

        REQUIRE(data.n_rows == test.n_rows);
        REQUIRE(data.n_cols == test.n_cols);

        REQUIRE(labels.n_elem == testLabels.n_elem);

        REQUIRE(arma::approx_equal(data, test, "absdiff", 0.0001));
        REQUIRE(arma::approx_equal(labels, testLabels, "absdiff", 0.0001));
    }
}

TEST_CASE("CreateMatrixHelperTest"){
    SECTION("Integer Test") {
        arma::mat baseMat = 
        {
            {1, 2, 3, 5},
            {4, 5, 6, 7}
        };
        std::vector<int> vec = {5, 6, 7, 1};
        std::variant<std::vector<int>, std::vector<bool>> vec_ = vec;
        arma::rowvec rowVec;
        createMatrixHelper(baseMat, rowVec, vec_);
        arma::mat test = 
        {
            {1, 2, 3, 5},
            {4, 5, 6, 7},
            {5, 6, 7, 1}
        };

        REQUIRE(baseMat.n_rows == test.n_rows);
        REQUIRE(baseMat.n_cols == test.n_cols);
    
        REQUIRE(arma::approx_equal(baseMat, test, "absdiff", 0.0001));

    }
    SECTION("BOOL TEST") {
        arma::mat baseMat = 
        {
            {1, 2, 3, 5},
            {4, 5, 6, 7}
        };
        std::vector<int> vec = {true, false, true, false};
        std::variant<std::vector<int>, std::vector<bool>> vec_ = vec;
        arma::rowvec rowVec;
        createMatrixHelper(baseMat, rowVec, vec_);
        arma::mat test = 
        {
            {1, 2, 3, 5},
            {4, 5, 6, 7},
            {1, 0, 1, 0}
        };

        REQUIRE(baseMat.n_rows == test.n_rows);
        REQUIRE(baseMat.n_cols == test.n_cols);
    
        REQUIRE(arma::approx_equal(baseMat, test, "absdiff", 0.0001));
    }
}