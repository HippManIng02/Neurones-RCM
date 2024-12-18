#include <stdint.h>
#include <iostream>
#include "Mlp.h"
#include "mnist_reader.h"


#define BEGIN 0
#define MAX_TRAIN 100
#define SIZE 784
#define HIDDEN_L_SIZE 128 

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

    //Initialisation du réseau de neurone : entrée(784); neurones(128); sorties(10)
    MLP mlp(SIZE, HIDDEN_L_SIZE, 10);

    std::cout<<"Entrainement du réseau de neurone."<<std::endl;
    //Propagation avant sur chaque image : forward propagation
    for(int i = 0; i < MAX_TRAIN; i++){
        //Convertion de l'image de uint8_t(0-255) à  un vecteur de float (0-1)
        std::vector<float> imageFormat(SIZE);
        for(int j =0; j < SIZE; j++ ){
            imageFormat[j] = images[i * SIZE + j] / 255.0f;
        }

        //Propagation avant
        std::vector<float> output = mlp.forward(imageFormat);
        //Récuppération de la valeur prédicte
        auto maxIt = std::max_element(output.begin(), output.end());
        int predictLabel = std::distance(output.begin(), maxIt);

        //Affichage de la valeur prédicte et de la valeur réel
        printf("Label réel=%d; Label prédict = %d\n ", *(labels+i), predictLabel);
        std::cout<<"*****************************************"<<std::endl;
     }
    
    free(images);
    free(labels);
    fclose(imageFile);
    fclose(labelFile);

    return EXIT_SUCCESS;
}