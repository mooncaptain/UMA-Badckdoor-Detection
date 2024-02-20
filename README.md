# UMA-Badckdoor-Detection

Dependencies for UMA_code:
python==3.8.10
numpy==1.24.4
torch==2.0.1
torchvision==0.15.2
opencv==4.8.0.74
Pillow==10.0.0
scikit-image== 0.21.0

	We provide our implementation of UMA on CIFAR 10 dataset:
1.Detect models with our method and print the anomaly index and transferability.
   -Place the models for detection in the directory: ./model_to_detect/ 
   -run the command: python UMA_detect.py
  We provide 6 models in the directory:./model_to_detect/ as examples, you can also train more backdoor-infected models using the provided codes in the directory:/train_backdoor_model/.
2.Train more backdoor-infected models
   We provide the codes to train varied trojaned models with using advanced triggers, including composite backdoor, filter backdoor, frequency domain backdoor, reflection backdoor and patch backdoor.
- the directory: ./train_backdoor_model/
-command: python3 train_backdoor_filter/frequency/patch/reflection/composite.py
3.Dataset
https://drive.google.com/file/d/1HWazTpbvgtYKKbY7hA-z_fYV-Rejh3CH/view?usp=sharing
https://drive.google.com/file/d/1itzy3dH2kvorBc2fHs4btqqe9bQY1u-4/view?usp=sharing
