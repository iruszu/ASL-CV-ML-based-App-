import Foundation
import CoreML

// Simple test to verify model loading
class ModelTest {
    static func testModelLoading() {
        print("🧪 Testing model loading...")
        
        // Test 1: Check if file exists in project directory
        let projectPath = "/Users/kellie/Desktop/Swift Apps/Apps/ASLApp/ASLApp/ASLClassifierModel.mlmodel"
        let fileManager = FileManager.default
        
        if fileManager.fileExists(atPath: projectPath) {
            print("✅ Model file exists in project directory")
        } else {
            print("❌ Model file NOT found in project directory")
        }
        
        // Test 2: Try to load model directly
        do {
            let model = try MLModel(contentsOf: URL(fileURLWithPath: projectPath))
            print("✅ Model can be loaded directly: \(model.modelDescription)")
        } catch {
            print("❌ Error loading model directly: \(error)")
        }
        
        // Test 3: Check bundle contents
        print("📁 Bundle resource path: \(Bundle.main.resourcePath ?? "nil")")
        
        if let bundlePath = Bundle.main.resourcePath {
            do {
                let contents = try fileManager.contentsOfDirectory(atPath: bundlePath)
                print("📄 All files in bundle:")
                for file in contents {
                    print("   - \(file)")
                }
            } catch {
                print("❌ Error listing bundle: \(error)")
            }
        }
    }
}

