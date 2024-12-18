#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include "Mlp.h"


MLP::MLP(int inputSize, int hiddenSize, int outputSize)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.01, 0.01);

    //Initialisation des poids et biais pour la couche cachée
    weights_input_hidden.resize(hiddenSize, std::vector<float>(inputSize));
    for(auto& row : weights_input_hidden){
        for(auto& weight : row){
            weight = dist(gen);
        }
    }

    bias_hidden.resize(hiddenSize, 0.0f);

    //Initialisation des poids et biais pour la couche de sortie
    weights_output_hidden.resize(outputSize, std::vector<float>(hiddenSize));
    for(auto& row : weights_output_hidden){
        for(auto& weight : row){
            weight = dist(gen);
        }
    }

    bias_output.resize(outputSize, 0.0f);
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
    std::vector<float> hiddenL;
    for(unsigned long i=0; i < weights_input_hidden.size(); i++){
        float z = bias_hidden[i];
        for(unsigned long j=0; j < inputs.size(); j++){
            z += inputs[j] * weights_input_hidden[i][j];
        }
        hiddenL.push_back(relu(z));
    }

    //Couche sortie
    std::vector<float> outputL;
    for(unsigned long i =0; i < weights_output_hidden.size(); i++){
        float z = bias_output[i];
        for(unsigned long j = 0; j < hiddenL.size(); j++){
            z += hiddenL[j] * weights_output_hidden[i][j];
        }
        outputL.push_back(z);
    }

    return softmax(outputL);
}