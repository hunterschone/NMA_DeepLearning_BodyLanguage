# NMA_DeepLearning_BodyLanguage
This code was used in our Neuromatch DeepLearning 2022 pod project. Pod contributors included: Jess, Linden, Hunter, Necdet, Ryan, Marufi, Abdullahi. Check out https://deeplearning.neuromatch.io/tutorials/intro.html

The aim of the project was to to test whether social interactions could be predicted from body language. We used the H2O: Human-to-Human-or-Object Interaction Dataset: https://kalisteo.cea.fr/wp-content/uploads/2021/12/README_H2O.html. This dataset has annotated images that include varirous social interactions, posture and motion of the subjects in the images. We trained *ResNet-5O* on these images. We tested the performance of the model when trained on only specific kinds of annotations (just the subject posture or motion or social interaction) in order to determine whether a model trained to predict posture or motion could perform well when classifying social interactions.

The project result slides have been uploaded as presentation_slides.pptx

To access the images and annotations we used for the project, compressed versions have been uploaded to https://osf.io/ah6rc/

\