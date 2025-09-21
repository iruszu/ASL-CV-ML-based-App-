# ASL App

A real-time American Sign Language (ASL) letter recognition iOS app built with
SwiftUI and Core ML. The app uses the device's camera to detect and classify ASL
hand signs, displaying the predicted letter with confidence scores.

## Features

- 📱 **Real-time ASL Recognition**: Live camera feed with instant letter
  classification
- 🤖 **Machine Learning Integration**: Uses Core ML and Vision frameworks for
  accurate predictions
- 📊 **Confidence Scoring**: Visual confidence indicators and percentage
  displays
- 🎯 **Multiple Model Support**: Handles both image-based and hand pose-based ML
  models
- 🔄 **Throttled Processing**: Optimized performance with intelligent frame
  processing
- 📱 **iOS Native**: Built with SwiftUI for a modern, responsive interface

## Technical Architecture

### Core Components

- **CameraManager**: Handles ML model loading, classification, and prediction
  processing
- **CameraView**: SwiftUI wrapper for UIKit camera integration
- **CameraPreviewView**: UIKit-based camera capture and video processing
- **ASLPrediction**: Data model for storing prediction results

### Machine Learning Pipeline

1. **Model Loading**: Automatically detects and loads the ASL classifier model
2. **Hand Detection**: Uses Vision framework for hand pose detection
3. **Feature Extraction**: Extracts hand landmarks and pose data
4. **Classification**: Runs predictions using Core ML
5. **Result Processing**: Converts model output to readable predictions

### Supported Model Types

- **Image-based Models**: Direct image classification using VNCoreMLModel
- **Hand Pose Models**: Landmark-based classification using
  VNDetectHumanHandPoseRequest
- **Multi-format Support**: Handles both `.mlmodel` and `.mlmodelc` files

## Requirements

- iOS 14.0+
- Xcode 12.0+
- Swift 5.0+
- Camera permission
- ASL classifier model file (`ASLClassifierModel.mlmodel`)

## Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd ASLApp
   ```

2. **Open in Xcode**:

   ```bash
   open ASLApp.xcodeproj
   ```

3. **Add your ML model**:

   - Place your `ASLClassifierModel.mlmodel` file in the project
   - Ensure it's added to the app target
   - The app will automatically detect and load the model

4. **Build and run**:
   - Select your target device or simulator
   - Press Cmd+R to build and run

## Usage

1. **Launch the app** and grant camera permission when prompted
2. **Position your hand** in front of the camera
3. **Make ASL signs** - the app will display:
   - Predicted letter
   - Confidence percentage
   - Visual confidence indicators
4. **View debug info** showing model status and frame count

## Model Requirements

The app expects an ASL classifier model with the following characteristics:

### For Image-based Models:

- Input: Image (any size, will be center-cropped)
- Output: Classification results with letter identifiers

### For Hand Pose Models:

- Input: Hand landmark data (multi-array format)
- Output: Letter predictions (A-Z)

### Model Training Considerations:

- Train on diverse hand positions and lighting conditions
- Include multiple angles and orientations
- Consider both left and right hands
- Ensure good contrast between hand and background

## Configuration

### Camera Settings

- **Camera Position**: Front-facing (configurable in `CameraPreviewView`)
- **Session Preset**: High quality
- **Processing Interval**: 0.5 seconds (throttled for performance)

### Classification Thresholds

- **Minimum Confidence**: 30% (configurable in `CameraManager`)
- **Visual Indicators**: 5-level confidence display

## Debug Features

The app includes comprehensive debugging capabilities:

- **Model Loading Status**: Real-time model load verification
- **Frame Processing Count**: Track processed frames
- **Bundle Contents**: Lists all available model files
- **Prediction Logging**: Detailed classification results
- **Error Handling**: Comprehensive error reporting

## Troubleshooting

### Common Issues

1. **Model Not Loading**:

   - Verify `ASLClassifierModel.mlmodel` is in the project
   - Check that the model is added to the app target
   - Review console logs for specific error messages

2. **No Predictions**:

   - Ensure good lighting conditions
   - Position hand clearly in camera view
   - Check confidence threshold settings
   - Verify model is properly trained

3. **Camera Issues**:
   - Grant camera permission in Settings
   - Check device camera functionality
   - Verify iOS version compatibility

### Debug Steps

1. Check console output for model loading messages
2. Verify frame processing is occurring
3. Test with different hand positions
4. Review confidence scores and thresholds

## Performance Optimization

- **Throttled Processing**: Limits classification to every 0.5 seconds
- **Background Processing**: Video processing on background queue
- **Memory Management**: Proper cleanup of camera resources
- **Efficient Model Loading**: Single model instance with reuse

## Future Enhancements

- [ ] Support for ASL words and phrases
- [ ] Multiple model selection
- [ ] Training data collection
- [ ] Offline mode improvements
- [ ] Accessibility features
- [ ] Custom confidence thresholds
- [ ] Model performance analytics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for
details.

## Acknowledgments

- Apple's Core ML and Vision frameworks
- SwiftUI for modern iOS development
- The ASL community for inspiration and feedback

---
