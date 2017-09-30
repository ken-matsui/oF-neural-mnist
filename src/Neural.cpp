#include "Neural.h"

Neural::Neural(){
    
    // Initialize weight.
    for (int i = 0; i < INPUT; ++i)
        for (int j = 0; j < HIDDEN/23; ++j)
            for (int n = 0; n < 23; ++n)
                input_hidden[n].coeffRef(i, j) = ofRandom(-1.0/sqrt(HIDDEN), 1.0/sqrt(HIDDEN));
    
    for (int i = 0; i < HIDDEN; ++i)
        for (int j = 0; j < OUTPUT; ++j)
            hidden_output.coeffRef(i, j) = ofRandom(-1.0/sqrt(OUTPUT), 1.0/sqrt(OUTPUT));

    output_one.setOnes();
    hidden_one.setOnes();
    
    ofEnableAlphaBlending();
    ofEnableBlendMode(OF_BLENDMODE_ALPHA);
    
    bondLine.setMode(OF_PRIMITIVE_LINES);
}

// sigmoid function
auto sigmoid = [](float x){ return (1/(1 + exp(-x))); };

// softmax function
void Neural::softmax(Matrix<float, 1, OUTPUT>* _out){
    
    // Acquire maximum value for overflow countermeasure.
    float&& C = _out->maxCoeff();
    for (int i = 0; i < OUTPUT; ++i)
        _out->coeffRef(0, i) = exp(_out->coeffRef(0, i) - C);
    for (int i = 0; i < OUTPUT; ++i)
        _out->coeffRef(0, i) /= _out->sum();
}

void Neural::draw(){
    // Draw Input layer.
    for (int i = 0; i < sqrt(INPUT); ++i){
        for (int j = 0; j < sqrt(INPUT); ++j){
            ofSetColor(ofMap(input.coeffRef(0, sqrt(INPUT) * i + j), 0.0, 1.0, 0, 255));
            ofDrawSphere(ofMap(i, 0, sqrt(INPUT) - 1, -500, 500), ofMap(j, 0, sqrt(INPUT) - 1, -500, 500), 1500, 10);
        }
    }
    
    // Calculate BondLine between Input layer and Hidden layer.
    for (int i = 0; i < sqrt(INPUT); ++i){
        for (int j = 0; j < sqrt(INPUT); ++j){
            // Calculate BondLine(start)
            bondLine.addVertex(ofVec3f(ofMap(i, 0, sqrt(INPUT) - 1, -500, 500), ofMap(j, 0, sqrt(INPUT) - 1, -500, 500), 1490));
            ofColor c;
            int hue = 0 + ofGetElapsedTimef() * 10;
            c.setHsb(hue % 255, 100, 100);
            bondLine.addColor(c);
            
            // Calculate BondLine(end)
            bondLine.addVertex(ofVec3f(ofMap(j, 0, sqrt(INPUT) - 1, -500, 500), ofMap(i, 0, sqrt(INPUT) - 1, -500, 500), 10));
            hue = 60 + ofGetElapsedTimef() * 10;
            c.setHsb(hue % 255, 100, 100);
            bondLine.addColor(c);
        }
    }
    
    // Draw Hidden layer.
    for (int i = 0; i < sqrt(HIDDEN); ++i){
        for (int j = 0; j < sqrt(HIDDEN); ++j){
            // TODO：正規化の入力範囲が不定
            ofSetColor(ofMap(hidden.coeffRef(0, sqrt(HIDDEN) * i + j), 0.0, 1.0, 0, 255));
            ofDrawSphere(ofMap(i, 0, sqrt(HIDDEN) - 1, -500, 500), ofMap(j, 0, sqrt(HIDDEN) - 1, -500, 500), 0, 10);
        }
    }
    
    // Calculate BondLine between Hidden layer and Output layer.
    for (int i = 0; i < sqrt(HIDDEN); ++i){
        for (int j = 0; j < sqrt(HIDDEN); ++j){
            ofSetColor(255, 255, 255, 1);
            // Calculate BondLine(start)
            bondLine.addVertex(ofVec3f(ofMap(i, 0, sqrt(HIDDEN) - 1, -500, 500), ofMap(j, 0, sqrt(HIDDEN) - 1, -500, 500), -10));
            ofColor c;
            int hue = 60 + ofGetElapsedTimef() * 10;
            c.setHsb(hue % 255, 100, 100);
            bondLine.addColor(c);
            
            // Calculate BondLine(end)
            bondLine.addVertex(ofVec3f(0, ofMap(i, 0, sqrt(HIDDEN) - 1, -300, 300), -1490));
            hue = 120 + ofGetElapsedTimef() * 10;
            c.setHsb(hue % 255, 100, 100);
            bondLine.addColor(c);
        }
    }
    
    // Draw Output layer.
    for (int i = 0; i < OUTPUT; ++i){
        ofSetColor(ofMap(output.coeffRef(0, i), 0.0, 1.0, 0, 255));
        ofDrawSphere(0, ofMap(i, 0, OUTPUT - 1, -300, 300), -1500, 10);
        
        ofSetColor(255);
        // Probability
        ofDrawBox(-ofMap(output.coeffRef(0, i), 0.0, 1.0, 0, 100/2)-20, ofMap(i, 0, OUTPUT - 1, -300, 300), -1500, ofMap(output.coeffRef(0, i), 0.0, 1.0, 0, 100), 10, 10);
        // Label
        ofDrawBitmapString(ofToString(i, 5), 20, ofMap(i, 0, OUTPUT - 1, -300, 300), -1500);
    }
    
    // Draw all BondLine.
    bondLine.draw();
    // Erase previous frame all BondLine.
    bondLine.clear();
}


void Neural::forword(ifstream ifs){
    
    string str;
    
    // Input layer = MNIST DATASET.
    for (int i = 0; i < INPUT; ++i){
        getline(ifs, str);
        // Normalized to 0.0 ~ 1.0
        input.coeffRef(0, i) = ofMap(std::stof(str), 0.0, 255.0, 0.0, 1.0);
    }
    // Hidden layer = sigmoid(Input layer * weight).
    for (int i = 0; i < 23; ++i){
        Matrix<float, 1, HIDDEN/23>&& temp = input * input_hidden[i];
        for (int j = 0; j < HIDDEN/23; ++j)
            hidden.coeffRef(0, (HIDDEN/23)*i + j) = sigmoid(temp.coeffRef(0, j));
    }
    // Output layer = softmax(Hidden layer * weight).
    output = hidden * hidden_output;
    softmax(&output);
}


void Neural::backword(const Matrix<float, 1, OUTPUT>& teacher){
    
    Matrix<float, 1, OUTPUT>&& output_error = teacher - output;

    Matrix<float, 1, HIDDEN>&& hidden_error = output_error * hidden_output.transpose();
    
    // Update weight.
    hidden_output += ETA * (hidden.transpose() * (output * output_error.transpose() * (output_one - output)));
    
    for (int i = 0; i < HIDDEN/23; ++i){
        Matrix<float, 1, HIDDEN/23>&& err = (hidden_one - hidden).block(0, 23*i, 1, 23);
        input_hidden[i] += ETA * (input.transpose() * (hidden * hidden_error.transpose() * err));
    }
}


void Neural::learn(ifstream ifs, const Matrix<float, 1, OUTPUT>& teacher){
    
    forword(move(ifs));
    backword(teacher);
}


void Neural::act(ifstream ifs){
    
    forword(move(ifs));
}


Neural::~Neural(){
    
}
