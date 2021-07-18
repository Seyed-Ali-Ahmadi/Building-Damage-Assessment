import sys
import torch
import torch.nn as nn
import functools
import numpy as np
import matplotlib.pyplot as plt
from networksForFeatureExtraction import ResnetFeatureExtractor9FeatureFromLayer23
from networksForFeatureExtraction import ResnetFeatureExtractor9FeatureFromLayer8
from networksForFeatureExtraction import ResnetFeatureExtractor9FeatureFromLayer10
from networksForFeatureExtraction import ResnetFeatureExtractor9FeatureFromLayer11
from networksForFeatureExtraction import ResnetFeatureExtractor9FeatureFromLayer5
from networksForFeatureExtraction import ResnetFeatureExtractor9FeatureFromLayer2
from skimage.transform import resize
from skimage import filters
from skimage import morphology
from skimage.io import imread
from saturateSomePercentile import saturateImage


def dcva(preImg, postImg, layers=[2, 5], feature=False):
    inputChannels = 3
    outputLayerNumbers = layers  # 2, 5, 8, 10, 11, 23
    # thresholdingStrategy = 'adaptive'  # adaptive, otsu, scaledOtsu
    thresholdingStrategy = 'otsu'  # adaptive, otsu, scaledOtsu
    otsuScalingFactor = 1.25
    objectMinSize = 100      # minimum size for objects in pixel
    topPercentSaturationOfImageOk = True
    topPercentToSaturate = 1

    nanVar = float('nan')

    # Defining parameters related to the CNN
    sizeReductionTable = [nanVar, nanVar, 1, nanVar, nanVar, 2, nanVar, nanVar, 4, nanVar, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                          nanVar, 2, nanVar, nanVar, 1, nanVar, nanVar, 1, 1]
    featurePercentileToDiscardTable = [nanVar, nanVar, 90, nanVar, nanVar, 90, nanVar, nanVar, 95, nanVar, 95, 95, 95,
                                       95, 95, 95, 95, 95, 95, nanVar, 95, nanVar, nanVar, 95, nanVar, nanVar, 0, 0]
    filterNumberTable = [nanVar, nanVar, 64, nanVar, nanVar, 128, nanVar, nanVar, 256, nanVar, 256, 256, 256, 256, 256,
                         256, 256, 256, 256, nanVar, 128, nanVar, nanVar, 64, nanVar, nanVar, 1, 1]

    preChangeImage = preImg
    postChangeImage = postImg

    # Pre-change and post-change image normalization
    if topPercentSaturationOfImageOk:
        preChangeImageNormalized = saturateImage().saturateSomePercentileMultispectral(preChangeImage,
                                                                                       topPercentToSaturate)
        postChangeImageNormalized = saturateImage().saturateSomePercentileMultispectral(postChangeImage,
                                                                                        topPercentToSaturate)
    # Reassigning pre-change and post-change image to normalized values
    data1 = np.copy(preChangeImageNormalized)
    data2 = np.copy(postChangeImageNormalized)

    # Checking image dimension
    imageSize = data1.shape
    imageSizeRow = imageSize[0]
    imageSizeCol = imageSize[1]
    imageNumberOfChannel = imageSize[2]

    # Initilizing net / model (G_B: acts as feature extractor here)
    input_nc = imageNumberOfChannel  # input number of channels
    output_nc = 6  # from Potsdam dataset number of classes
    ngf = 64  # number of gen filters in first conv layer
    norm_layer = nn.BatchNorm2d
    use_dropout = False

    netForFeatureExtractionLayer23 = ResnetFeatureExtractor9FeatureFromLayer23(input_nc, output_nc, ngf, norm_layer,
                                                                               use_dropout, 9)
    netForFeatureExtractionLayer11 = ResnetFeatureExtractor9FeatureFromLayer11(input_nc, output_nc, ngf, norm_layer,
                                                                               use_dropout, 9)
    netForFeatureExtractionLayer10 = ResnetFeatureExtractor9FeatureFromLayer10(input_nc, output_nc, ngf, norm_layer,
                                                                               use_dropout, 9)
    netForFeatureExtractionLayer8 = ResnetFeatureExtractor9FeatureFromLayer8(input_nc, output_nc, ngf, norm_layer,
                                                                             use_dropout, 9)
    netForFeatureExtractionLayer5 = ResnetFeatureExtractor9FeatureFromLayer5(input_nc, output_nc, ngf, norm_layer,
                                                                             use_dropout, 9)
    netForFeatureExtractionLayer2 = ResnetFeatureExtractor9FeatureFromLayer2(input_nc, output_nc, ngf, norm_layer,
                                                                             use_dropout, 9)

    state_dict = torch.load('./trainedNet/RGB/trainedModelFinal', map_location='cpu')
    # for name, param in state_dict.items():
    #    print(name)

    netForFeatureExtractionLayer23Dict = netForFeatureExtractionLayer23.state_dict()
    state_dictForLayer23 = state_dict
    state_dictForLayer23 = {k: v for k, v in netForFeatureExtractionLayer23Dict.items() if k in state_dictForLayer23}

    netForFeatureExtractionLayer11Dict = netForFeatureExtractionLayer11.state_dict()
    state_dictForLayer11 = state_dict
    state_dictForLayer11 = {k: v for k, v in netForFeatureExtractionLayer11Dict.items() if k in state_dictForLayer11}

    netForFeatureExtractionLayer10Dict = netForFeatureExtractionLayer10.state_dict()
    state_dictForLayer10 = state_dict
    state_dictForLayer10 = {k: v for k, v in netForFeatureExtractionLayer10Dict.items() if k in state_dictForLayer10}

    netForFeatureExtractionLayer8Dict = netForFeatureExtractionLayer8.state_dict()
    state_dictForLayer8 = state_dict
    state_dictForLayer8 = {k: v for k, v in netForFeatureExtractionLayer8Dict.items() if k in state_dictForLayer8}

    netForFeatureExtractionLayer5Dict = netForFeatureExtractionLayer5.state_dict()
    state_dictForLayer5 = state_dict
    state_dictForLayer5 = {k: v for k, v in netForFeatureExtractionLayer5Dict.items() if k in state_dictForLayer5}

    netForFeatureExtractionLayer2Dict = netForFeatureExtractionLayer2.state_dict()
    state_dictForLayer2 = state_dict
    state_dictForLayer2 = {k: v for k, v in netForFeatureExtractionLayer2Dict.items() if k in state_dictForLayer2}

    netForFeatureExtractionLayer23.load_state_dict(state_dictForLayer23)
    netForFeatureExtractionLayer11.load_state_dict(state_dictForLayer11)
    netForFeatureExtractionLayer10.load_state_dict(state_dictForLayer10)
    netForFeatureExtractionLayer8.load_state_dict(state_dictForLayer8)
    netForFeatureExtractionLayer5.load_state_dict(state_dictForLayer5)
    netForFeatureExtractionLayer2.load_state_dict(state_dictForLayer2)

    input_nc = imageNumberOfChannel  # input number of channels
    output_nc = imageNumberOfChannel  # output number of channels
    ngf = 64  # number of gen filters in first conv layer
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    use_dropout = False

    # changing all nets to eval mode
    netForFeatureExtractionLayer23.eval()
    netForFeatureExtractionLayer23.requires_grad = False

    netForFeatureExtractionLayer11.eval()
    netForFeatureExtractionLayer11.requires_grad = False

    netForFeatureExtractionLayer10.eval()
    netForFeatureExtractionLayer10.requires_grad = False

    netForFeatureExtractionLayer8.eval()
    netForFeatureExtractionLayer8.requires_grad = False

    netForFeatureExtractionLayer5.eval()
    netForFeatureExtractionLayer5.requires_grad = False

    netForFeatureExtractionLayer2.eval()
    netForFeatureExtractionLayer2.requires_grad = False

    torch.no_grad()

    eachPatch = imageSizeRow
    numImageSplitRow = imageSizeRow / eachPatch
    numImageSplitCol = imageSizeCol / eachPatch
    cutY = list(range(0, imageSizeRow, eachPatch))
    cutX = list(range(0, imageSizeCol, eachPatch))
    additionalPatchPixel = 64

    layerWiseFeatureExtractorFunction = [nanVar, nanVar, netForFeatureExtractionLayer2, nanVar, nanVar,
                                         netForFeatureExtractionLayer5, nanVar, nanVar, netForFeatureExtractionLayer8,
                                         nanVar, netForFeatureExtractionLayer10, netForFeatureExtractionLayer11, nanVar,
                                         nanVar, nanVar, nanVar, nanVar, nanVar, nanVar, nanVar, nanVar, nanVar, nanVar,
                                         netForFeatureExtractionLayer23, nanVar, nanVar, nanVar, nanVar]

    # Extracting bi-temporal features
    modelInputMean = 0.406
    for outputLayerIter in range(0, len(outputLayerNumbers)):
        outputLayerNumber = outputLayerNumbers[outputLayerIter]
        filterNumberForOutputLayer = filterNumberTable[outputLayerNumber]
        featurePercentileToDiscard = featurePercentileToDiscardTable[outputLayerNumber]
        featureNumberToRetain = int(np.floor(filterNumberForOutputLayer * ((100 - featurePercentileToDiscard) / 100)))
        sizeReductionForOutputLayer = sizeReductionTable[outputLayerNumber]
        patchOffsetFactor = int(additionalPatchPixel / sizeReductionForOutputLayer)
        print('Processing layer number:' + str(outputLayerNumber))

        timeVector1Feature = np.zeros([imageSizeRow, imageSizeCol, filterNumberForOutputLayer])
        timeVector2Feature = np.zeros([imageSizeRow, imageSizeCol, filterNumberForOutputLayer])
        for kY in range(0, len(cutY)):
            for kX in range(0, len(cutX)):

                # extracting subset of image 1
                if kY == 0 and kX == 0:
                    patchToProcessDate1 = data1[cutY[kY]:(cutY[kY] + eachPatch + additionalPatchPixel),
                                                cutX[kX]:(cutX[kX] + eachPatch + additionalPatchPixel), :]
                elif kY == 0:
                    patchToProcessDate1 = data1[cutY[kY]:(cutY[kY] + eachPatch + additionalPatchPixel),
                                                (cutX[kX] - additionalPatchPixel):(cutX[kX] + eachPatch), :]
                elif kX == 0:
                    patchToProcessDate1 = data1[(cutY[kY] - additionalPatchPixel):
                                                (cutY[kY] + eachPatch),
                                                cutX[kX]:(cutX[kX] + eachPatch + additionalPatchPixel), :]
                else:
                    patchToProcessDate1 = data1[(cutY[kY] - additionalPatchPixel):
                                                (cutY[kY] + eachPatch),
                                                (cutX[kX] - additionalPatchPixel):(cutX[kX] + eachPatch), :]
                # extracting subset of image 2
                if kY == 0 and kX == 0:
                    patchToProcessDate2 = data2[cutY[kY]:(cutY[kY] + eachPatch + additionalPatchPixel),
                                                cutX[kX]:(cutX[kX] + eachPatch + additionalPatchPixel), :]
                elif kY == 0:
                    patchToProcessDate2 = data2[cutY[kY]:(cutY[kY] + eachPatch + additionalPatchPixel),
                                                (cutX[kX] - additionalPatchPixel):(cutX[kX] + eachPatch), :]
                elif kX == 0:
                    patchToProcessDate2 = data2[(cutY[kY] - additionalPatchPixel):
                                                (cutY[kY] + eachPatch),
                                                cutX[kX]:(cutX[kX] + eachPatch + additionalPatchPixel), :]
                else:
                    patchToProcessDate2 = data2[(cutY[kY] - additionalPatchPixel):
                                                (cutY[kY] + eachPatch),
                                                (cutX[kX] - additionalPatchPixel):(cutX[kX] + eachPatch), :]

                # converting to pytorch varibales and changing dimension for input to net
                patchToProcessDate1 = patchToProcessDate1 - modelInputMean

                inputToNetDate1 = torch.from_numpy(patchToProcessDate1)
                inputToNetDate1 = inputToNetDate1.float()
                inputToNetDate1 = np.swapaxes(inputToNetDate1, 0, 2)
                inputToNetDate1 = np.swapaxes(inputToNetDate1, 1, 2)
                inputToNetDate1 = inputToNetDate1.unsqueeze(0)

                patchToProcessDate2 = patchToProcessDate2 - modelInputMean

                inputToNetDate2 = torch.from_numpy(patchToProcessDate2)
                inputToNetDate2 = inputToNetDate2.float()
                inputToNetDate2 = np.swapaxes(inputToNetDate2, 0, 2)
                inputToNetDate2 = np.swapaxes(inputToNetDate2, 1, 2)
                inputToNetDate2 = inputToNetDate2.unsqueeze(0)

                # running model on image 1 and converting features to numpy format

                with torch.no_grad():
                    obtainedFeatureVals1 = layerWiseFeatureExtractorFunction[outputLayerNumber](inputToNetDate1)
                obtainedFeatureVals1 = obtainedFeatureVals1.squeeze()
                obtainedFeatureVals1 = obtainedFeatureVals1.data.numpy()

                # running model on image 2 and converting features to numpy format
                with torch.no_grad():
                    obtainedFeatureVals2 = layerWiseFeatureExtractorFunction[outputLayerNumber](inputToNetDate2)
                obtainedFeatureVals2 = obtainedFeatureVals2.squeeze()
                obtainedFeatureVals2 = obtainedFeatureVals2.data.numpy()
                # this features are in format (filterNumber, sizeRow, sizeCol)

                ##clipping values to +1 to -1 range, be careful, if network is changed, maybe we need to modify this
                obtainedFeatureVals1 = np.clip(obtainedFeatureVals1, -1, +1)
                obtainedFeatureVals2 = np.clip(obtainedFeatureVals2, -1, +1)

                # obtaining features from image 1: resizing and truncating additionalPatchPixel
                if kY == 0 and kX == 0:
                    for processingFeatureIter in range(0, filterNumberForOutputLayer):
                        timeVector1Feature[cutY[kY]:(cutY[kY] + eachPatch),
                        cutX[kX]:(cutX[kX] + eachPatch), processingFeatureIter] = \
                            resize(obtainedFeatureVals1[processingFeatureIter,
                                   0:int(eachPatch / sizeReductionForOutputLayer),
                                   0:int(eachPatch / sizeReductionForOutputLayer)],
                                   (eachPatch, eachPatch))

                elif kY == 0:
                    for processingFeatureIter in range(0, filterNumberForOutputLayer):
                        timeVector1Feature[cutY[kY]:(cutY[kY] + eachPatch),
                        cutX[kX]:(cutX[kX] + eachPatch), processingFeatureIter] = \
                            resize(obtainedFeatureVals1[processingFeatureIter,
                                   0:int(eachPatch / sizeReductionForOutputLayer),
                                   (patchOffsetFactor):
                                   (int(eachPatch / sizeReductionForOutputLayer) + patchOffsetFactor)],
                                   (eachPatch, eachPatch))
                elif kX == 0:
                    for processingFeatureIter in range(0, filterNumberForOutputLayer):
                        timeVector1Feature[cutY[kY]:(cutY[kY] + eachPatch),
                        cutX[kX]:(cutX[kX] + eachPatch), processingFeatureIter] = \
                            resize(obtainedFeatureVals1[processingFeatureIter,
                                   (patchOffsetFactor):
                                   (int(eachPatch / sizeReductionForOutputLayer) + patchOffsetFactor),
                                   0:int(eachPatch / sizeReductionForOutputLayer)],
                                   (eachPatch, eachPatch))
                else:
                    for processingFeatureIter in range(0, filterNumberForOutputLayer):
                        timeVector1Feature[cutY[kY]:(cutY[kY] + eachPatch),
                        cutX[kX]:(cutX[kX] + eachPatch), processingFeatureIter] = \
                            resize(obtainedFeatureVals1[processingFeatureIter,
                                   (patchOffsetFactor):
                                   (int(eachPatch / sizeReductionForOutputLayer) + patchOffsetFactor),
                                   (patchOffsetFactor):
                                   (int(eachPatch / sizeReductionForOutputLayer) + patchOffsetFactor)],
                                   (eachPatch, eachPatch))
                # obtaining features from image 2: resizing and truncating additionalPatchPixel
                if kY == 0 and kX == 0:
                    for processingFeatureIter in range(0, filterNumberForOutputLayer):
                        timeVector2Feature[cutY[kY]:(cutY[kY] + eachPatch),
                        cutX[kX]:(cutX[kX] + eachPatch), processingFeatureIter] = \
                            resize(obtainedFeatureVals2[processingFeatureIter,
                                   0:int(eachPatch / sizeReductionForOutputLayer),
                                   0:int(eachPatch / sizeReductionForOutputLayer)],
                                   (eachPatch, eachPatch))

                elif kY == 0:
                    for processingFeatureIter in range(0, filterNumberForOutputLayer):
                        timeVector2Feature[cutY[kY]:(cutY[kY] + eachPatch),
                        cutX[kX]:(cutX[kX] + eachPatch), processingFeatureIter] = \
                            resize(obtainedFeatureVals2[processingFeatureIter,
                                   0:int(eachPatch / sizeReductionForOutputLayer),
                                   (patchOffsetFactor):
                                   (int(eachPatch / sizeReductionForOutputLayer) + patchOffsetFactor)],
                                   (eachPatch, eachPatch))
                elif kX == 0:
                    for processingFeatureIter in range(0, filterNumberForOutputLayer):
                        timeVector2Feature[cutY[kY]:(cutY[kY] + eachPatch),
                        cutX[kX]:(cutX[kX] + eachPatch), processingFeatureIter] = \
                            resize(obtainedFeatureVals2[processingFeatureIter,
                                   (patchOffsetFactor):
                                   (int(eachPatch / sizeReductionForOutputLayer) + patchOffsetFactor),
                                   0:int(eachPatch / sizeReductionForOutputLayer)],
                                   (eachPatch, eachPatch))
                else:
                    for processingFeatureIter in range(0, filterNumberForOutputLayer):
                        timeVector2Feature[cutY[kY]:(cutY[kY] + eachPatch),
                        cutX[kX]:(cutX[kX] + eachPatch), processingFeatureIter] = \
                            resize(obtainedFeatureVals2[processingFeatureIter,
                                   (patchOffsetFactor):
                                   (int(eachPatch / sizeReductionForOutputLayer) + patchOffsetFactor),
                                   (patchOffsetFactor):
                                   (int(eachPatch / sizeReductionForOutputLayer) + patchOffsetFactor)],
                                   (eachPatch, eachPatch))

        timeVectorDifferenceMatrix = timeVector1Feature - timeVector2Feature

        nonZeroVector = []
        stepSizeForStdCalculation = int(imageSizeRow / 2)
        for featureSelectionIter1 in range(0, imageSizeRow, stepSizeForStdCalculation):
            for featureSelectionIter2 in range(0, imageSizeCol, stepSizeForStdCalculation):
                timeVectorDifferenceSelectedRegion = timeVectorDifferenceMatrix \
                    [featureSelectionIter1:(featureSelectionIter1 + stepSizeForStdCalculation), \
                                                     featureSelectionIter2:(
                                                             featureSelectionIter2 + stepSizeForStdCalculation),
                                                     0:filterNumberForOutputLayer]
                stdVectorDifferenceSelectedRegion = np.std(timeVectorDifferenceSelectedRegion, axis=(0, 1))
                featuresOrderedPerStd = np.argsort(
                    -stdVectorDifferenceSelectedRegion)  # negated array to get argsort result in descending order
                nonZeroVectorSelectedRegion = featuresOrderedPerStd[0:featureNumberToRetain]
                nonZeroVector = np.union1d(nonZeroVector, nonZeroVectorSelectedRegion)

        modifiedTimeVector1 = timeVector1Feature[:, :, nonZeroVector.astype(int)]
        modifiedTimeVector2 = timeVector2Feature[:, :, nonZeroVector.astype(int)]

        ##Normalize the features (separate for both images)
        meanVectorsTime1Image = np.mean(modifiedTimeVector1, axis=(0, 1))
        stdVectorsTime1Image = np.std(modifiedTimeVector1, axis=(0, 1))
        normalizedModifiedTimeVector1 = (modifiedTimeVector1 - meanVectorsTime1Image) / stdVectorsTime1Image

        meanVectorsTime2Image = np.mean(modifiedTimeVector2, axis=(0, 1))
        stdVectorsTime2Image = np.std(modifiedTimeVector2, axis=(0, 1))
        normalizedModifiedTimeVector2 = (modifiedTimeVector2 - meanVectorsTime2Image) / stdVectorsTime2Image

        # feature aggregation across channels
        if outputLayerIter == 0:
            timeVector1FeatureAggregated = np.copy(normalizedModifiedTimeVector1)
            timeVector2FeatureAggregated = np.copy(normalizedModifiedTimeVector2)
        else:
            timeVector1FeatureAggregated = np.concatenate((timeVector1FeatureAggregated, normalizedModifiedTimeVector1),
                                                          axis=2)
            timeVector2FeatureAggregated = np.concatenate((timeVector2FeatureAggregated, normalizedModifiedTimeVector2),
                                                          axis=2)

    del obtainedFeatureVals1, obtainedFeatureVals2, timeVector1Feature, timeVector2Feature, inputToNetDate1, inputToNetDate2
    del netForFeatureExtractionLayer5, netForFeatureExtractionLayer8, netForFeatureExtractionLayer10, netForFeatureExtractionLayer11, netForFeatureExtractionLayer23

    absoluteModifiedTimeVectorDifference = np.absolute(
        saturateImage().saturateSomePercentileMultispectral(timeVector1FeatureAggregated, 5) -
        saturateImage().saturateSomePercentileMultispectral(timeVector2FeatureAggregated, 5))

    # take absolute value for binary CD
    detectedChangeMap = np.linalg.norm(absoluteModifiedTimeVectorDifference, axis=2)
    detectedChangeMapNormalized = (detectedChangeMap - np.amin(detectedChangeMap)) / (
            np.amax(detectedChangeMap) - np.amin(detectedChangeMap))

    cdMap = np.zeros(detectedChangeMapNormalized.shape, dtype=bool)
    if thresholdingStrategy == 'adaptive':
        for sigma in range(101, 202, 50):
            adaptiveThreshold = 2 * filters.gaussian(detectedChangeMapNormalized, sigma)
            cdMapTemp = (detectedChangeMapNormalized > adaptiveThreshold)
            cdMapTemp = morphology.remove_small_objects(cdMapTemp, min_size=objectMinSize)
            cdMap = cdMap | cdMapTemp
    elif thresholdingStrategy == 'otsu':
        otsuThreshold = filters.threshold_otsu(detectedChangeMapNormalized)
        cdMap = (detectedChangeMapNormalized > otsuThreshold)
        cdMap = morphology.remove_small_objects(cdMap, min_size=objectMinSize)
    elif thresholdingStrategy == 'scaledOtsu':
        otsuThreshold = filters.threshold_otsu(detectedChangeMapNormalized)
        cdMap = (detectedChangeMapNormalized > otsuScalingFactor * otsuThreshold)
        cdMap = morphology.remove_small_objects(cdMap, min_size=objectMinSize)
    else:
        sys.exit('Unknown thresholding strategy')
    cdMap = morphology.binary_closing(cdMap, morphology.disk(3))

    if feature:
        finalChangeMap = detectedChangeMapNormalized
    else:
        finalChangeMap = cdMap

    return finalChangeMap


# ------------------------------------------------------------------------------
# path = 'D:/00.University/PhD Thesis Implementation/thesisEnv/dataset/train/images/'
# preChangeImage = imread(path + 'hurricane-florence_00000048_pre_disaster.png')
# postChangeImage = imread(path + 'hurricane-florence_00000048_post_disaster.png')
#
# cdmap, cdfeature = dcva(preChangeImage, postChangeImage, layers=[2], feature=True)
#
# from skimage.color import rgb2gray
# cdmap = rgb2gray(postChangeImage) + cdmap * 0.3
#
# plt.figure()
# plt.subplot(221), plt.imshow(preChangeImage), plt.xticks([]), plt.yticks([]), plt.title('pre image')
# plt.subplot(222), plt.imshow(postChangeImage), plt.xticks([]), plt.yticks([]), plt.title('post image')
# plt.subplot(223), plt.imshow(cdfeature, cmap='gray'), plt.xticks([]), plt.yticks([])
# plt.subplot(224), plt.imshow(cdmap, cmap='gray'), plt.xticks([]), plt.yticks([])
# plt.tight_layout()
# plt.show()


# # ---------------------------------------------------------------------------
# from skimage.color import rgb2gray
# import pickle
#
# database = pickle.load(open("databaseDictionary.pkl", "rb"))
# disaster_type = 'mixed'
# damage_class = 'destroyed'
#
# print('Checking files for ' + damage_class.upper() + ' in ' + disaster_type.upper() + ' ...')
# files = database[disaster_type][damage_class]
# for idx, row in files.iterrows():
#     file = row['img_name']
#     postFile = file.replace('labels', 'images') + '.png'
#     preFile = postFile.replace('post', 'pre')
#
#     postChangeImage = imread(postFile)
#     preChangeImage = imread(preFile)
#
#     cdmap, cdfeature = dcva(preChangeImage, postChangeImage, layers=[2], feature=True)
#     cdmap = rgb2gray(postChangeImage) + cdmap * 0.2
#
#     plt.figure()
#     plt.suptitle(file.split('/')[-1])
#     plt.subplot(221), plt.imshow(rgb2gray(preChangeImage), cmap='gray')
#     plt.xticks([]), plt.yticks([]), plt.title('pre image')
#     plt.subplot(222), plt.imshow(np.abs(rgb2gray(postChangeImage)-rgb2gray(preChangeImage)))
#     plt.xticks([]), plt.yticks([]), plt.title('post image')
#     plt.subplot(223), plt.imshow(cdfeature), plt.xticks([])
#     plt.yticks([]), plt.title('change features')
#     plt.subplot(224), plt.imshow(cdmap, cmap='gray')
#     plt.xticks([]), plt.yticks([]), plt.title('changes')
#     plt.tight_layout()
#     plt.show()

