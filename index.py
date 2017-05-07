import colordescriptor
import argparse
import glob
import cv2
import os.path
import zernikemoments

def main():
    #Contsruct the argumen parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required = True,
                     help = "Path to the directory that contains the images to be indexed")
    ap.add_argument("-i", "--index", required = True,
            help = "Path to where the computed index will be stored")
    args = vars(ap.parse_args())

    #initialize the color descriptor
    cd = colordescriptor.ColorDescriptor((8, 12, 3))
    zm = zernikemoments.ZernikeMoments(25)

    #open the output index file for writing
    cdoutput = open(os.path.join(args["index"],'colordescriptor.csv'), "w")
    zmoutput = open(os.path.join(args["index"],'zernikemoment.csv'), "w")

    #use glob to grab the image paths and loop over them
    for imagePath in glob.glob(args["dataset"] + "/*.jpg"):
        #extract the image ID from the image path and load the image itself
        # imageID = imagePath[imagePath.rfind("/")+ 1:]
        imageID = os.path.abspath(imagePath)
        image = cv2.imread(imagePath)

        #describe the image with colordescriptor
        features = cd.describe(image)
        write_features(features, cdoutput, imageID)

        #wdescribe the image with zernikemoments
        features = zm.describe(image)
        write_features(features, zmoutput, imageID)

    cdoutput.close()
    zmoutput.close()
                        
def write_features(feature, filename, imageID):
    features = [str(f) for f in feature]
    filename.write("%s,%s\n" % (imageID, ",".join(features)))

if __name__ == '__main__':
    main()
