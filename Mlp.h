#ifndef  __MLP__H__
#define __MLP__H__

#include <vector>
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>

class MLP{
private:
    uint inputSize;
    uint hiddenSize;
    uint outputSize;
    // poids et biais de la couche d'entrée à la couche cachée
    float *weights_input_hidden;
    float *bias_hidden;
    // Poids et biais de la couche cachée à la couche de sortie
    float *weights_output_hidden;
    float *bias_output;


public:
    //Constructeur
    MLP(uint inputSize, uint hiddenSize, uint outputSize = 10);
    //Destructeur
    ~MLP();

    //Propagation avant
    std::vector<float> forward(const std::vector<float>& inputs);

    //Fonction d'activation pour la couche cachée
    float relu(float x) const;

    //Fonction d'ativation softmax pour la couche de sortie
    std::vector<float> softmax(const std::vector<float>& z) const;

};

#endif // ! __MLP__H__
