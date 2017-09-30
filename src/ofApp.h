#pragma once

#include "ofMain.h"
#include "Neural.h"
#include <memory.h>

using namespace std;

class ofApp : public ofBaseApp{
    
public:
    void setup();
    void update();
    void draw();
    
    void keyPressed(int key);
    void keyReleased(int key);
    void mouseMoved(int x, int y );
    void mouseDragged(int x, int y, int button);
    void mousePressed(int x, int y, int button);
    void mouseReleased(int x, int y, int button);
    void mouseEntered(int x, int y);
    void mouseExited(int x, int y);
    void windowResized(int w, int h);
    void dragEvent(ofDragInfo dragInfo);
    void gotMessage(ofMessage msg);
    
    void exit();
    
    
    // Camera
    ofCamera cam;
    ofVec3f camPos;
    ofVec3f camLook;
    float theta { PI };
    float theta2 { PI };
    bool camWork { false };
    bool camWave { false };
    bool stop { false };
    
    // File
    string directory;
    int whichFile { 0 };
    bool changeFile { false };
    int aggregate[10] = { 0 };
    
    // Neural Network
    bool learn { false };
    Eigen::Matrix<float, 1, Neural::OUTPUT> teacher;
    Neural* neural;
};
