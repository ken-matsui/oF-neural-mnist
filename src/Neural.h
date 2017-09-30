#pragma once

#include "ofMain.h"
#include "Eigen/Core"

using namespace Eigen;
using namespace std;

class Neural {
    
public:
    static constexpr int INPUT = 28*28; // 784, 28*28
    static constexpr int OUTPUT = 10; // 10, 0~9
    static constexpr int HIDDEN = (INPUT + OUTPUT) * (2.0/3.0); // 529, 23*23
    
    // Implicit type conversion prohibited.
    explicit Neural();
    // Copy & Move prohibited.
    Neural(const Neural&) = delete;
    Neural& operator=(const Neural&) = delete;
    
    ~Neural();

    void draw();
    void learn(ifstream ifs, const Matrix<float, 1, OUTPUT>& teacher);
    void act(ifstream ifs);
    
    
private:
    const float ETA = 0.01; // Learning coefficient
    
    
    void softmax(Matrix<float, 1, OUTPUT>* _output);
    
    void forword(ifstream ifs);
    void backword(const Matrix<float, 1, OUTPUT>& teacher);
    
    
    // Input layer
    Matrix<float, 1, INPUT> input;
    
    // Weight between Input layer and Hidden layer
    // Since TooBigError is issued, it divides the matrix.
    //Matrix<float, INPUT_NUM, HIDDEN_NUM> input_hidden;
    Matrix<float, INPUT, HIDDEN/23> input_hidden[23];
    
    // Hidden layer
    Matrix<float, 1, HIDDEN> hidden;
    
    // Weight between Hidden layer and Output layer
    Matrix<float, HIDDEN, OUTPUT> hidden_output;
    
    // Output layer
    Matrix<float, 1, OUTPUT> output;
    
    // All one Matrix
    Matrix<float, 1, OUTPUT> output_one;
    Matrix<float, 1, HIDDEN> hidden_one;
    
    ofMesh bondLine;
};
