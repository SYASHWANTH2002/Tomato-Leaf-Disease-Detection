Tomato Leaf Disease Detection Using Deep  Learning

This project provides an end-to-end deep learning solution for agricultural diagnostics, specifically focusing on the high-accuracy classification of 10 different tomato leaf diseases. By integrating modern computer vision architectures with reliability tools, it offers a "production-ready" framework that not only predicts diseases but also explains its reasoning and quantifies its own certainty.

1. The Dataset & Preparation
The foundation of the project is the Unified Tomato Dataset, which encompasses a wide variety of pathological conditions.  
Disease Coverage: The model identifies 10 categories, including Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Mosaic virus, Yellow Leaf Curl Virus, and healthy leaves.  
Data Handling (tomato_dataset.py): A custom pipeline handles automated class mapping to fix naming inconsistencies across different data sources.  
Pre-processing (transforms.py): Images are resized to 224x224 and normalized using standard ImageNet statistics.  
Augmentation: To improve generalization, training involves random horizontal flips, rotations, color jittering, and the addition of Gaussian noise.

2. Model Architecture 
The "brain" of the project is a Hybrid Transformer-CNN model designed to overcome the limitations of using a single architecture.  
Local Feature Extraction: ConvNeXt-Tiny is utilized to capture fine-grained textures and small lesion patterns on the leaves.  
Global Context Extraction: EfficientNet-B3 runs in parallel to understand the broader structure and shape of the leaf.  
Feature Alignment: Since these backbones produce different data shapes, projection layers are used to align them into a unified 512-dimensional space.  
Transformer Fusion: A Multi-Head Self-Attention module fuses these features, allowing the model to focus on the most critical parts of the image (the "attention" mechanism).  
Final Classification: A fully connected head with 40% Dropout produces the final disease prediction.

3. Training Strategy 
Training was conducted with a focus on stability and handling class imbalances.
Optimizer: The system uses AdamW with a learning rate of 5e-5.
Class Weights: The loss function is weighted based on the frequency of each class to ensure that rare diseases are not ignored by the model.
Convergence: As shown in the training logs, the model reaches a best loss of approximately 0.1814 within 15 epochs.

4. Evaluation & Results
The model delivers exceptional performance metrics on the primary test set.  Overall Accuracy: The system achieved a final accuracy of 98.76%.  Per-Class Precision: Most classes show precision and recall above 0.95, with the Yellow Leaf Curl Virus reaching an F1-score of 0.9982.  Confusion Matrix: Automated generation of confusion matrices helps identify which diseases look most similar to the model (e.g., distinguishing between different types of blights).

5. Advanced Reliability Tests
Standard accuracy is rarely enough for real-world deployment, so the project includes three advanced stress tests:
A. Robustness Testing (robustness.png)
This test measures how well the model performs when the image quality is poor.  
Gaussian Noise: 98.60% accuracy.  
Gaussian Blur: 98.32% accuracy.  
Lighting Changes: Maintained over 98.7% accuracy despite brightness and contrast shifts.  
B. Cross-Dataset Testing (cross dataset.png)
This evaluates the model on completely external data it has never seen before.  
Accuracy: The model maintained a solid 78.47% accuracy on external data, which is highly impressive for a cross-domain test.  
C. Few-Shot Learning (few shot.png)
This measures the model's ability to learn from very few examples.  
Performance: Even with only 5 images per class, the model achieved an accuracy of 82.00%.

6. Explainability & Uncertainty
To ensure users can trust the model's decisions, two final layers were added:Uncertainty Estimation (uncertainty.py): Using Monte Carlo (MC) Dropout, the model provides an average confidence score of 0.9366 and an uncertainty score of 0.1804. Low-confidence images are automatically rejected to prevent false diagnoses.  Visual Explanation (gradcam.py): Generates heatmaps (overlaid on the original leaf) that show the user exactly which spots or lesions the model used to make its decision.  
