import SwiftUI
import AVFoundation
import CoreML
import Vision
import VisionKit

// Data model for predictions
struct ASLPrediction {
    let letter: String
    let confidence: Float
}

// Camera manager to handle model integration
class CameraManager: ObservableObject {
    @Published var currentPrediction: ASLPrediction?
    @Published var isModelLoaded: Bool = false
    @Published var frameCount: Int = 0
    
    private var model: VNCoreMLModel?
    private var classificationRequest: VNCoreMLRequest?
    private var mlModel: MLModel?
    private var handPoseRequest: VNDetectHumanHandPoseRequest?
    
    init() {
        // Run model test first
        ModelTest.testModelLoading()
        loadModel()
    }
    
    private func loadModel() {
        print("🔍 Starting model loading...")
        
        // Debug: List all files in bundle
        print("📁 Bundle contents:")
        if let bundlePath = Bundle.main.resourcePath {
            let fileManager = FileManager.default
            do {
                let contents = try fileManager.contentsOfDirectory(atPath: bundlePath)
                for file in contents {
                    if file.contains("mlmodel") || file.contains("ML") {
                        print("   📄 Found: \(file)")
                    }
                }
            } catch {
                print("❌ Error listing bundle contents: \(error)")
            }
        }
        
        // Try multiple loading methods
        var modelURL: URL?
        
        // Method 1: Try .mlmodel
        if let url = Bundle.main.url(forResource: "ASLClassifierModel", withExtension: "mlmodel") {
            modelURL = url
            print("✅ Found .mlmodel file at: \(url)")
        }
        // Method 2: Try .mlmodelc (compiled)
        else if let url = Bundle.main.url(forResource: "ASLClassifierModel", withExtension: "mlmodelc") {
            modelURL = url
            print("✅ Found .mlmodelc file at: \(url)")
        }
        // Method 3: Try without extension
        else if let url = Bundle.main.url(forResource: "ASLClassifierModel", withExtension: nil) {
            modelURL = url
            print("✅ Found model file (no extension) at: \(url)")
        }
        // Method 4: Search for any mlmodel file
        else {
            print("🔍 Searching for any .mlmodel files...")
            if let bundlePath = Bundle.main.resourcePath {
                let fileManager = FileManager.default
                do {
                    let contents = try fileManager.contentsOfDirectory(atPath: bundlePath)
                    for file in contents {
                        if file.hasSuffix(".mlmodel") || file.hasSuffix(".mlmodelc") {
                            let url = URL(fileURLWithPath: bundlePath).appendingPathComponent(file)
                            modelURL = url
                            print("✅ Found model file: \(file) at: \(url)")
                            break
                        }
                    }
                } catch {
                    print("❌ Error searching for model files: \(error)")
                }
            }
        }
        
        guard let finalModelURL = modelURL else {
            print("❌ Could not find any model file in bundle")
            print("💡 Make sure ASLClassifierModel.mlmodel is added to your Xcode project target")
            return
        }
        
        print("✅ Using model at: \(finalModelURL)")
        
        do {
            let mlModel = try MLModel(contentsOf: finalModelURL)
            print("✅ MLModel loaded successfully")
            
            // Check model input requirements
            let modelDescription = mlModel.modelDescription
            print("📋 Model input features:")
            for (name, feature) in modelDescription.inputDescriptionsByName {
                print("   - \(name): \(feature.type)")
            }
            
            print("📋 Model output features:")
            for (name, feature) in modelDescription.outputDescriptionsByName {
                print("   - \(name): \(feature.type)")
            }
            
            // Try different approaches based on model type
            if modelDescription.inputDescriptionsByName.values.contains(where: { $0.type == .image }) {
                // Image-based model
                print("🖼️ Detected image-based model")
                let model = try VNCoreMLModel(for: mlModel)
                self.model = model
                
                let request = VNCoreMLRequest(model: model) { [weak self] request, error in
                    if let error = error {
                        print("❌ Classification error: \(error)")
                    }
                    self?.processClassificationResults(request.results)
                }
                request.imageCropAndScaleOption = .centerCrop
                self.classificationRequest = request
                
            } else {
                // Hand pose or video-based model - use direct MLModel
                print("🤚 Detected hand pose/video-based model")
                self.mlModel = mlModel
                
                // Create a custom classification method for hand pose models
                self.setupHandPoseClassification()
            }
            
            print("✅ Model setup completed successfully")
            
            DispatchQueue.main.async {
                self.isModelLoaded = true
            }
            
        } catch {
            print("❌ Error loading model: \(error)")
        }
    }
    
    private func setupHandPoseClassification() {
        print("🤚 Setting up hand pose classification...")
        
        // Create hand pose detection request
        let handPoseRequest = VNDetectHumanHandPoseRequest { [weak self] request, error in
            if let error = error {
                print("❌ Hand pose detection error: \(error)")
                return
            }
            
            guard let observations = request.results as? [VNHumanHandPoseObservation] else {
                print("❌ No hand pose observations")
                return
            }
            
            // Process the first detected hand
            if let handObservation = observations.first {
                self?.processHandPoseObservation(handObservation)
            }
        }
        
        // Configure the request
        handPoseRequest.maximumHandCount = 1
        self.handPoseRequest = handPoseRequest
        
        print("✅ Hand pose detection request created")
    }
    
    private func processHandPoseObservation(_ handObservation: VNHumanHandPoseObservation) {
        print("🤚 Processing hand pose observation...")
        
        guard let mlModel = mlModel else {
            print("❌ No ML model available")
            return
        }
        
        do {
            // Extract hand landmarks and convert to the format expected by your model
            let poses = try extractHandPoses(from: handObservation)
            
            // Create input for the model
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "poses": MLFeatureValue(multiArray: poses)
            ])
            
            // Run prediction
            let prediction = try mlModel.prediction(from: input)
            processHandPosePrediction(prediction)
            
        } catch {
            print("❌ Error processing hand pose: \(error)")
        }
    }
    
    private func extractHandPoses(from handObservation: VNHumanHandPoseObservation) throws -> MLMultiArray {
        // Get all available hand landmarks
        let availableJoints = VNHumanHandPoseObservation.allHandLandmarks
        
        // Create a multi-array to store the pose data
        // Assuming your model expects a specific format - adjust as needed
        let poseArray = try MLMultiArray(shape: [1, NSNumber(value: availableJoints.count), NSNumber(value: 3)], dataType: .float32)
        
        var index = 0
        for joint in availableJoints {
            do {
                let point = try handObservation.landmark(for: joint)
                let normalizedPoint = VNImagePointForNormalizedPoint(point.location, Int(handObservation.imageSize.width), Int(handObservation.imageSize.height))
                
                // Store x, y, z coordinates
                poseArray[index * 3] = NSNumber(value: normalizedPoint.x)
                poseArray[index * 3 + 1] = NSNumber(value: normalizedPoint.y)
                poseArray[index * 3 + 2] = NSNumber(value: point.confidence)
                
                index += 1
            } catch {
                // If landmark is not available, use zeros
                poseArray[index * 3] = NSNumber(value: 0.0)
                poseArray[index * 3 + 1] = NSNumber(value: 0.0)
                poseArray[index * 3 + 2] = NSNumber(value: 0.0)
                index += 1
            }
        }
        
        print("✅ Extracted \(availableJoints.count) hand landmarks")
        return poseArray
    }
    
    private func processClassificationResults(_ results: [VNObservation]?) {
        print("🔍 Processing classification results...")
        
        guard let results = results as? [VNClassificationObservation] else { 
            print("❌ No classification results or wrong type")
            return 
        }
        
        print("✅ Got \(results.count) classification results")
        
        // Get the top prediction
        if let topPrediction = results.first {
            let confidence = topPrediction.confidence
            let identifier = topPrediction.identifier
            
            print("🎯 Top prediction: \(identifier) with confidence: \(confidence)")
            
            // Show all predictions for debugging
            for (index, result) in results.prefix(3).enumerated() {
                print("   \(index + 1). \(result.identifier): \(result.confidence)")
            }
            
            // Only update if confidence is above threshold
            if confidence > 0.3 { // Lowered threshold for debugging
                print("✅ Updating UI with prediction: \(identifier)")
                DispatchQueue.main.async {
                    self.currentPrediction = ASLPrediction(
                        letter: identifier,
                        confidence: confidence
                    )
                }
            } else {
                print("⚠️ Confidence too low: \(confidence) < 0.3")
            }
        }
    }
    
    func classifyFrame(_ pixelBuffer: CVPixelBuffer) {
        print("🔄 Running classification on frame...")
        
        DispatchQueue.main.async {
            self.frameCount += 1
        }
        
        // Check if we have a Vision-based model or direct MLModel
        if let request = classificationRequest {
            // Image-based model using Vision framework
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
            
            do {
                try handler.perform([request])
                print("✅ Vision classification request performed successfully")
            } catch {
                print("❌ Error performing Vision classification: \(error)")
            }
        } else if let handPoseRequest = handPoseRequest {
            // Hand pose-based model using Vision hand pose detection
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
            
            do {
                try handler.perform([handPoseRequest])
                print("✅ Hand pose detection request performed successfully")
            } catch {
                print("❌ Error performing hand pose detection: \(error)")
            }
        } else {
            print("❌ No model available for classification")
        }
    }
    
    
    private func processHandPosePrediction(_ prediction: MLFeatureProvider) {
        print("🔍 Processing hand pose prediction...")
        
        // Get the output features
        let outputName = prediction.featureNames.first ?? "output"
        print("📝 Output name: \(outputName)")
        
        if let output = prediction.featureValue(for: outputName) {
            print("📊 Output type: \(output.type)")
            
            // Handle different output types
            switch output.type {
            case .multiArray:
                if let multiArray = output.multiArrayValue {
                    processMultiArrayOutput(multiArray)
                }
            case .dictionary:
                let dictionary = output.dictionaryValue
                processDictionaryOutput(dictionary)
            default:
                print("⚠️ Unsupported output type: \(output.type)")
            }
        }
    }
    
    private func processMultiArrayOutput(_ multiArray: MLMultiArray) {
        print("📊 Processing multi-array output with \(multiArray.count) elements")
        
        // Find the index with the highest value (most likely prediction)
        var maxIndex = 0
        var maxValue = Float.leastNormalMagnitude
        
        for i in 0..<multiArray.count {
            let value = multiArray[i].floatValue
            if value > maxValue {
                maxValue = value
                maxIndex = i
            }
        }
        
        // Convert index to letter (assuming A=0, B=1, etc.)
        let letter = String(UnicodeScalar(65 + maxIndex)!) // A=65 in ASCII
        let confidence = maxValue
        
        print("🎯 Predicted: \(letter) with confidence: \(confidence)")
        
        if confidence > 0.3 {
            DispatchQueue.main.async {
                self.currentPrediction = ASLPrediction(
                    letter: letter,
                    confidence: confidence
                )
            }
        }
    }
    
    private func processDictionaryOutput(_ dictionary: [AnyHashable: NSNumber]) {
        print("📊 Processing dictionary output with \(dictionary.count) keys")
        
        // Find the key with the highest value
        var maxKey = ""
        var maxValue = Float.leastNormalMagnitude
        
        for (key, value) in dictionary {
            let floatValue = value.floatValue
            if floatValue > maxValue {
                maxValue = floatValue
                maxKey = String(describing: key)
            }
        }
        
        print("🎯 Predicted: \(maxKey) with confidence: \(maxValue)")
        
        if maxValue > 0.3 {
            DispatchQueue.main.async {
                self.currentPrediction = ASLPrediction(
                    letter: maxKey,
                    confidence: maxValue
                )
            }
        }
    }
}

// Main SwiftUI View
struct ContentView: View {
    @StateObject private var cameraManager = CameraManager()
    
    var body: some View {
        VStack {
            CameraView(cameraManager: cameraManager)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color.black)
            
            // Debug info
            VStack {
                Text("Model Loaded: \(cameraManager.isModelLoaded ? "✅" : "❌")")
                    .foregroundColor(.white)
                Text("Frames Processed: \(cameraManager.frameCount)")
                    .foregroundColor(.white)
            }
            .padding()
            .background(Color.black.opacity(0.7))
            .cornerRadius(10)
            
            // Display classification results
            if let prediction = cameraManager.currentPrediction {
                VStack {
                    Text("Predicted Letter:")
                        .font(.headline)
                        .foregroundColor(.white)
                    
                    Text(prediction.letter)
                        .font(.system(size: 72, weight: .bold))
                        .foregroundColor(.yellow)
                    
                    Text("Confidence: \(Int(prediction.confidence * 100))%")
                        .font(.subheadline)
                        .foregroundColor(.white)
                    
                    // Confidence indicator
                    HStack {
                        ForEach(0..<5) { index in
                            Circle()
                                .fill(index < Int(prediction.confidence * 5) ? Color.green : Color.gray)
                                .frame(width: 10, height: 10)
                        }
                    }
                }
                .padding()
                .background(Color.black.opacity(0.7))
                .cornerRadius(10)
            }
        }
    }
}

// SwiftUI wrapper for UIKit camera view
struct CameraView: UIViewRepresentable {
    let cameraManager: CameraManager
    
    func makeUIView(context: Context) -> CameraPreviewView {
        let cameraView = CameraPreviewView()
        cameraView.cameraManager = cameraManager
        
        // Request camera permission and setup
        AVCaptureDevice.requestAccess(for: .video) { granted in
            DispatchQueue.main.async {
                if granted {
                    cameraView.setupCamera()
                } else {
                    print("Camera permission denied")
                }
            }
        }
        
        return cameraView
    }
    
    func updateUIView(_ uiView: CameraPreviewView, context: Context) {
        // Updates go here if needed
    }
}

// UIKit view that handles the actual camera
class CameraPreviewView: UIView {
    private var captureSession: AVCaptureSession?
    private var previewLayer: AVCaptureVideoPreviewLayer?
    private var videoOutput: AVCaptureVideoDataOutput?
    
    // Reference to camera manager for classification
    var cameraManager: CameraManager?
    
    // Throttling for classification
    private var lastClassificationTime: CFTimeInterval = 0
    private let classificationInterval: CFTimeInterval = 0.5 // Classify every 0.5 seconds
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .black
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        backgroundColor = .black
    }
    
    func setupCamera() {
        // Create capture session
        let captureSession = AVCaptureSession()
        captureSession.sessionPreset = .high
        
        // Get the front camera (change to .back if you want rear camera)
        guard let camera = AVCaptureDevice.default(
            .builtInWideAngleCamera,
            for: .video,
            position: .front
        ) else {
            print("Unable to access camera")
            return
        }
        
        do {
            // Create input from camera
            let input = try AVCaptureDeviceInput(device: camera)
            
            if captureSession.canAddInput(input) {
                captureSession.addInput(input)
            }
            
            // Create video output for frame processing
            let videoOutput = AVCaptureVideoDataOutput()
            videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
            
            if captureSession.canAddOutput(videoOutput) {
                captureSession.addOutput(videoOutput)
            }
            
            // Create preview layer
            let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
            previewLayer.videoGravity = .resizeAspectFill
            
            // Add preview layer to view
            layer.addSublayer(previewLayer)
            
            // Store references
            self.captureSession = captureSession
            self.previewLayer = previewLayer
            self.videoOutput = videoOutput
            
            // Start the session on background thread
            DispatchQueue.global(qos: .userInitiated).async {
                captureSession.startRunning()
            }
            
        } catch {
            print("Error setting up camera: \(error.localizedDescription)")
        }
    }
    
    override func layoutSubviews() {
        super.layoutSubviews()
        // Update preview layer frame when view bounds change
        previewLayer?.frame = bounds
    }
    
    // Clean up when view disappears
    func stopCamera() {
        captureSession?.stopRunning()
    }
}

// Handle video frames and run classification
extension CameraPreviewView: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // Throttle classification to avoid overwhelming the system
        let currentTime = CACurrentMediaTime()
        guard currentTime - lastClassificationTime >= classificationInterval else { return }
        
        lastClassificationTime = currentTime
        
        // Convert sample buffer to pixel buffer
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { 
            print("❌ Could not get pixel buffer from sample buffer")
            return 
        }
        
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        print("📷 Processing frame: \(width)x\(height)")
        
        // Run classification
        cameraManager?.classifyFrame(pixelBuffer)
    }
}

#Preview {
    ContentView()
}
