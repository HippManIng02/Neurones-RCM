#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include "Mlp.h"

MLP::~MLP(){
    delete[] weights_input_hidden;
    delete[] weights_output_hidden;
    delete[] bias_hidden;
    delete[] bias_output;
}

MLP::MLP(uint inputSize, uint hiddenSize, uint outputSize)
: inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.01, 0.01);

    //Allocation des poids et des biais
    weights_input_hidden = new float[hiddenSize * inputSize];
    weights_output_hidden = new float[outputSize * hiddenSize];
    bias_hidden = new float[hiddenSize];
    bias_output = new float[outputSize];

    //Initialisation des poids et biais pour la couche cachée
    for(uint i=0; i< hiddenSize; i++){
        bias_hidden[i] = 0.0f;
        for(uint j=0;j<inputSize; j++){
            weights_input_hidden[i*inputSize +j] = dist(gen);
        }
    }

    //Initialisation des poids et biais pour la couche de sortie
    for(uint i=0; i< outputSize; i++){
        bias_output[i] = 0.0f;
        for(uint j=0;j<hiddenSize; j++){
            weights_output_hidden[i*hiddenSize +j] = dist(gen);
        }
    }
   
}

//Fonction d'activation ReLu
float MLP::relu(float x) const{
    return std::max(0.0f, x);
}


//Fonction d'activation softmax
std::vector<float> MLP::softmax(const std::vector<float>& z) const{
    float sum =0.0f;
    for(float val : z){
        sum += exp(val);
    }

    std::vector<float> proba;
    for(float val: z){
        proba.push_back(exp(val) / sum);
    }

    return proba;
}



//Propagation avant : forward propagation
std::vector<float> MLP::forward(const std::vector<float>& inputs)
{
    //couche cachée
    std::vector<float> hiddenL(hiddenSize, 0.0f);
    for(unsigned long i=0; i < hiddenSize; i++){
        float z = bias_hidden[i];
        for(unsigned long j=0; j < inputs.size(); j++){
            z += inputs[j] * weights_input_hidden[i * inputSize + j];
        }
        hiddenL[i] = relu(z);
    }

    //Couche sortie
    std::vector<float> outputL(outputSize, 0.0f);;
    for(unsigned long i =0; i < outputSize; i++){
        float z = bias_output[i];
        for(unsigned long j = 0; j < hiddenSize; j++){
            z += hiddenL[j] * weights_output_hidden[i * hiddenSize + j];
        }
        outputL[i] = z;
    }

    return softmax(outputL);
}


//Propagation avant : forward propagation
std::vector<float> MLP::forward(const std::vector<float>& inputs, std::vector<float>& hiddenL)
{
    //couche cachée
    hiddenL.resize(hiddenSize, 0.0f);
    for(unsigned long i=0; i < hiddenSize; i++){
        float z = bias_hidden[i];
        for(unsigned long j=0; j < inputs.size(); j++){
            z += inputs[j] * weights_input_hidden[i * inputSize + j];
        }
        hiddenL[i] = relu(z);
    }

    //Couche sortie
    std::vector<float> outputL(outputSize, 0.0f);;
    for(unsigned long i =0; i < outputSize; i++){
        float z = bias_output[i];
        for(unsigned long j = 0; j < hiddenSize; j++){
            z += hiddenL[j] * weights_output_hidden[i * hiddenSize + j];
        }
        outputL[i] = z;
    }

    return softmax(outputL);
}

//Entrainement du réseau de neurone sur un certain nombre d'époche donné
void MLP::train(const std::vector<std::vector<float>>& trainingData, const std::vector<int>& labels, float learningRate, int epochs){
    for(int epoch =0; epoch < epochs; epoch ++){
        float totalLoss = 0.0f;
        for(size_t sampleIdx = 0; sampleIdx < trainingData.size(); sampleIdx++){
            //récupérer une image d'entrainement avec son label
            const std::vector<float>& inputs = trainingData[sampleIdx];
            unsigned long label = labels[sampleIdx];

            // Étape 1: Propagation avant (on utilise la fonction forward)
            std::vector<float> hiddenL;
            std::vector<float> predictedValues = forward(inputs, hiddenL);

            /*
            *Calcul de la fonction perte (Simplication de la fonction de perte car le label est un vecteur composé en majorité de zéro et une seule valeur significatif 1)
            *  Fonction initiale L=−k=1∑N (​yk​⋅log(y^​k​)) qui devient L=−log(y^​label​)
            */
            totalLoss -= std::log(predictedValues[label]);

            // Étape 2 : La backpropagation
            //Gradient pour la couche de sortie
            std::vector<float> outputGradient(outputSize, 0.0f);
            for(unsigned long i = 0; i < outputSize; i++){
                outputGradient[i] = predictedValues[i] - (i == label ? 1.0f : 0.0f);
            }

            //Gradient pour les poids de la couche sortie
            for(unsigned long i = 0; i < outputSize; i++){
                for(unsigned long j = 0; j < hiddenSize; j++){
                    weights_output_hidden[i * hiddenSize +j] -= learningRate * outputGradient[i] * hiddenL[j];
                }
                bias_output[i] -= learningRate * outputGradient[i];
            }

            //Gradient pour la couche cachée(1 seul couche cachée)
            std::vector<float> hiddenGradient(hiddenSize, 0.0f);
            for(unsigned long i = 0; i < hiddenSize; i++){
                float sum = 0.0f;
                for(unsigned long j = 0; j < outputSize; j++){
                    sum += outputGradient[j] * weights_output_hidden[j * hiddenSize + i];
                }
                hiddenGradient[i] = sum * ((hiddenL[i] > 0 ) ? 1.0f : 0.0f); //Relu derivative
            }

            //Gradient pour le poids de la couche cachée
            for(unsigned long i = 0; i < hiddenSize ; i++){
                for(unsigned long j = 0; j < outputSize ; j++){
                    weights_input_hidden[i * inputSize + j] -= learningRate * hiddenGradient[i] * inputs[j]; 
                }
                bias_hidden[i] -= learningRate * hiddenGradient[i];
            }
        }

        std::cout << "Epoch " << epoch + 1 << " : Perte moyenne = " << totalLoss / trainingData.size() << std::endl;
    }
}

