#include <stdint.h>
#include <iostream>
#include "Mlp.h"
#include "mnist_reader.h"


#define BEGIN 0
#define MAX_TRAIN 1000
#define SIZE 784
#define HIDDEN_L_SIZE 128
#define OUTPUT_SIZE 10
#define TRAININGRATE 0.001
#define EPOCHS 50

int main(){
    //Ouverture des fichiers MNIST (images et labels)
    FILE* imageFile = fopen("mnist/train-images-idx3-ubyte", "r");
    FILE* labelFile = fopen("mnist/train-labels-idx1-ubyte", "r");
    
    if (imageFile == NULL){
        fprintf(stderr, "Erreur : Impossible d'ouvrir le fichier image MNIST. \n");
        return EXIT_FAILURE;
    }

    if (labelFile == NULL){
        fprintf(stderr, "Erreur : Impossible d'ouvrir le fichier label MNIST. \n");
        return EXIT_FAILURE;
    }

    //Lire MAX_TRAIN images et labels à partir de la BEGIN image
    uint8_t * images =  readMnistImages(imageFile, BEGIN, MAX_TRAIN);
    uint8_t * labels =  readMnistLabels(labelFile, BEGIN, MAX_TRAIN);

     if (images == NULL){
        fprintf(stderr, "Erreur : Impossible de lire les images. \n");
        fclose(imageFile);
        fclose(labelFile);
        return EXIT_FAILURE;
    }

    if (labels == NULL){
        fprintf(stderr, "Erreur : Impossible de lire les labels. \n");
        fclose(imageFile);
        fclose(labelFile);
        return EXIT_FAILURE;
    }

    // Les données d'entraînement
    std::vector<std::vector<float>> trainingData(MAX_TRAIN, std::vector<float>(SIZE));
    std::vector<int> trainingLabels(MAX_TRAIN);

    for(int i = 0; i < MAX_TRAIN; i++){
        for(int j = 0; j < SIZE; j++){
            trainingData[i][j] = images[i* SIZE + j] / 255.0f;//Convertion de l'image de uint8_t(0-255) à  un vecteur de float (0-1)
        }
        trainingLabels[i] = labels[i];
    }
    
    free(images);
    free(labels);
    fclose(imageFile);
    fclose(labelFile);

    //Initialisation du réseau de neurone : entrée(784); neurones(128); sorties(10)
    MLP mlp(SIZE, HIDDEN_L_SIZE, OUTPUT_SIZE);

    std::cout<<"Entrainement du réseau de neurone."<<std::endl;
    mlp.train(trainingData, trainingLabels, TRAININGRATE, EPOCHS);

    std::cout<< "Test de prédiction après entrainement."<<std::endl;
    for(int i=0; i < 10; i++){
        std::vector<float> output = mlp.forward(trainingData[i]);
        //Récuppération de la valeur prédicte
        auto maxIt = std::max_element(output.begin(), output.end());
        int predictLabel = std::distance(output.begin(), maxIt);

        //Affichage de la valeur prédicte et de la valeur réel
        printf("Label réel=%d; Label prédict = %d\n ", trainingLabels[i], predictLabel);
        std::cout<<"*****************************************"<<std::endl;
    }

    return EXIT_SUCCESS;
}