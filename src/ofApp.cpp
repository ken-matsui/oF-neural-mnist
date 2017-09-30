#include "ofApp.h"

void ofApp::setup(){
    
    // Setup the screen.
    ofSetVerticalSync(true);
    glEnable(GL_DEPTH_TEST);
    ofSetFrameRate(60);
    ofBackground(20);
    
    // Setup sphere.
    ofNoFill();
    
    // Setup camera.
    camPos.set(0, 0, 6000); // 2500
    camLook.set(0, 5, 0);
    
    // Setup directory.
    directory = "/Users/ken/Documents/openFrameworks/apps/myApps/numberRecognition/bin/data/";
    
    // Instance.
    neural = new Neural();
}

void ofApp::update(){
    
    if(learn == true) {
        // Input a text as image.
        string fileName = directory + to_string(whichFile) + ".txt";
        ifstream ifs(fileName);
        string str;
        if(ifs.fail()) {
            cerr << "File do not exist.\n";
            std::exit(0);
        }
        // Input a "TeacherNumber".
        getline(ifs, str);
        if(changeFile == false) {
            ++aggregate[stoi(str)];
            // one-hot expression.
            teacher.setZero();
            teacher.coeffRef(0, stoi(str)) = 1.0;
        }
        changeFile = true;
        
        // Start learn.
        neural->learn(move(ifs), teacher);
        
        // Camera animation.
        if(camPos.x == 0 && camPos.z > -2000 && stop == false){
            if(camWork == false) { // 1
                camPos.z -= 4;
            }
            else if(camWave ==false) { // 5
                camPos.x = 1000 * cos(theta);
                camPos.y = 1000 * sin(theta);
                
                camLook.x = 2000 * cos(theta);
                
                theta -= 0.005;
            }
            else { // 7(Last)
                camPos.z = 2000 * sin(theta2);
                camPos.y = -2000 * sin(theta2 * 2.0);
                
                camLook.y = 0;
                camLook.x = 50;
                
                //theta -= 0.005;
                theta2 += 0.005;
                if(theta2 > 11)
                    stop = true;
            }
        }
        else if(theta > -PI && stop == false) { // 2
            camPos.x = 2000 * sin(theta);
            camPos.z = 2000 * cos(theta);
            
            theta -= 0.005;
        }
        else if(camPos.z < 2500 && stop == false) {
            if(camWork == false) { // 3
                camPos.z += 4;
                camLook.x = 0;
            }
            else { // 6
                camPos.z += 4;
                
                camLook.x = 2000 * cos(theta);
                camLook.z = 2000 * sin(theta);
                
                theta -= 0.008;
                
                camWave = true;
            }
        }
        else if(stop == false) { // 4
            camPos.x = 0.0;
            camWork = true;
            theta = 0;
        }
        
        // File scaning
        if(whichFile < 9999) {
            whichFile++;
            changeFile = false;
        }
    }
    
    // Setup camera.
    cam.setPosition(camPos);
    cam.lookAt(camLook, ofVec3f(0, 1, 0));
    
    // Setup the fps and fileLabel.
    int fps = ceil(ofGetFrameRate());
    ofSetWindowTitle("fps : "+ofToString(fps)/*+", FileLabel : "+str*/);
}

void ofApp::draw(){
    
    ofPushMatrix();
    cam.begin();
    // Adjust display.
    ofRotateZ(-90);
    x
    neural->draw();
    
    cam.end();
    ofPopMatrix();
    
    // Log.
    /*for (int i = 0; i < 10; ++i){
     ofSetColor(255, 255, 255);
     string info = ofToString(i, 2) + " -> " + ofToString(aggregate[i], 2);
     ofDrawBitmapString(info, 30, 30 + i*30);
     }*/
}

void ofApp::keyPressed(int key){
    switch (key) {
        case OF_KEY_RETURN:
            // Start learn.
            // TODO：学習アルゴリズムが未完．
            learn = true;
            break;
        case 'a':
            // Start act.
            // TODO：実行部分が未定．
        case 's':
            // Save weight.
        case 'l':
            // Load weight.
        case 'f':
            // Full screen.
            ofToggleFullscreen();
            break;
        default:
            break;
    }
}

void ofApp::keyReleased(int key){
    
}

void ofApp::mouseMoved(int x, int y ){
    
}

void ofApp::mouseDragged(int x, int y, int button){
    
}

void ofApp::mousePressed(int x, int y, int button){
    
}

void ofApp::mouseReleased(int x, int y, int button){
    
}

void ofApp::mouseEntered(int x, int y){
    
}

void ofApp::mouseExited(int x, int y){
    
}

void ofApp::windowResized(int w, int h){
    
}

void ofApp::gotMessage(ofMessage msg){
    
}

void ofApp::dragEvent(ofDragInfo dragInfo){
    
}

void ofApp::exit(){
    delete neural;
}
