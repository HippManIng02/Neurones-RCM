#ifndef  __MLP__H__
#define __MLP__H__

#include <vector>
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>

class MLP{
private:
    // poids et biais de la couche d'entrée à la couche cachée
    std::vector<std::vector<float>> weights_input_hidden;
    std::vector<float> bias_hidden;

    // Poids et biais de la couche cachée à la couche de sortie
    std::vector<std::vector<float>> weights_output_hidden;
    std::vector<float> bias_output;


public:
    //Constructeur
    MLP(int inputSize, int hiddenSize, int outputSize = 10);

    //Propagation avant
    std::vector<float> forward(const std::vector<float>& inputs);

    //Fonction d'activation pour la couche cachée
    float relu(float x) const;

    //Fonction d'ativation softmax pour la couche de sortie
    std::vector<float> softmax(const std::vector<float>& z) const;

};

#endif // ! __MLP__H__
