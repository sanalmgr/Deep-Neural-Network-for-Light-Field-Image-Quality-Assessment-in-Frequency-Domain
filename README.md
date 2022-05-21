# Deep-Neural-Network-for-Light-Field-Image-Quality-Assessment-in-Frequency-Domain
In this work, we present a novel no-reference light field image quality assessment (LF-IQA) method, which is based on a Deep Neural Network that uses Frequency domain inputs (DNNF-LFIQA). The proposed method predicts the quality of an LF image by taking as input the Fourier magnitude spectrum of LF contents, represented as horizontal and vertical epipolar-plane images (EPIs). Specifically, DNNF-LFIQA is composed of two processing streams (stream1 and stream2) that take as inputs the horizontal and the vertical epipolar plane images in the frequency domain. Both streams are composed of identical blocks of convolutional neural networks (CNNs), with their outputs being combined using two fusion blocks. Finally, the fused feature vector is fed to a regression block to generate the quality prediction.

## Code:
## Training Model:
1. Prepare the horizontal and vertical EPIs using the method MultiEPL https://bit.ly/3Da8fB6.
2. To train the model, import functions from fft_train_model.py file, and pass the parameters accordingly.
