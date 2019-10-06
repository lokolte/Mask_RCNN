from custom import ObjectInference
import matplotlib.pyplot as plt

if __name__ == '__main__':

    FLAG_TRAINING=False # Change to false for training
    SHOW_IMAGE_FLAG=True # Change to false to not display mask

    modelToTrain = ObjectInference()

    # Train a new model starting from pre-trained COCO weights
    # dataset=/path/to/balloon/dataset weights=coco

    # Resume training a model that you had trained earlier
    # dataset=/path/to/balloon/dataset weights=last

    # Train a new model starting from ImageNet weights
    # dataset=/path/to/balloon/dataset weights=imagenet

    if FLAG_TRAINING:
        modelToTrain.trainModel(dataset='/home/alforro/TigoCampusParty/Mask_RCNN/cedula/images', weights='last')

    # Apply color splash to an image
    # weights=/path/to/weights/file.h5 image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    # weights=last video=<URL or path to file>

    is_cedula, splashed_image = modelToTrain.splashModel(weights='last', image='/home/lokolte/Proyectos/git/Mask_RCNN/cedula/images/splash/cedula_1.jpeg')

    if is_cedula:
        print("Esto es una cedula. Deteccion exitosa.")
    else:
        print("Esto no es una cedula. Deteccion fallida.")

    if SHOW_IMAGE_FLAG:
        plt.imshow(splashed_image)
        plt.show()
