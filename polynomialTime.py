import pygame
from pygame.locals import *
import pygame_textinput
import sys
import cv2
import numpy as np
import string
import matplotlib.pyplot as plt
import pylab
import operator
import os
from fractions import Fraction
import copy


# taken from 15-112 course page
def almostEqual(d1, d2, epsilon=10**-7):
    # note: use math.isclose() outside 15-112 with Python version 3.5 or later
    return (abs(d2 - d1) < epsilon)


# wrote for 15-112 week1 practice
def distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


# truncates a float n to number of places specified
def truncate(n, places):
    a = str(n)
    return float(a[:places + 1])


# i edited these sources to work with my data and openCV framework and to work with my desire to recognize math and expoonents
# i did however, modify this function to recognize exponents with spacing
# ie 6x6 would show 36 but 6x^6 would graph it
# credit for ml and image manipulation goes to these sources
# machine learning credit:
# https://www.youtube.com/watch?v=c96w1JS28AY (links to github in video)
# https://github.com/MicrocontrollersAndMore/OpenCV_KNN_Character_Recognition_Machine_Learning/blob/master/train_and_test.py
MIN_CONTOUR_AREA = 100
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

class ContourWithData():

    npaContour = None
    boundingRect = None
    intRectX = 0
    intRectY = 0
    intRectWidth = 0
    intRectHeight = 0
    fltArea = 0.0

    def calculateRectTopLeftPointAndWidthAndHeight(self):
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):
        if self.fltArea < MIN_CONTOUR_AREA: return False
        return True


def main(img):
    allContoursWithData = []
    validContoursWithData = []

    npaClassifications = np.loadtxt("classificationsNew.txt", np.float32)
    npaFlattenedImages = np.loadtxt("flattened_imagesNew.txt", np.float32)

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
    kNearest = cv2.ml.KNearest_create()
    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
    imgTestingNumbers = cv2.imread(img)

    imgGray = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255,                                 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       
                                      cv2.THRESH_BINARY_INV, 11, 2)                                    

    imgThreshCopy = imgThresh.copy()
    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,
                                                 cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_SIMPLE)

    for npaContour in npaContours:
        contourWithData = ContourWithData()
        contourWithData.npaContour = npaContour
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)
        allContoursWithData.append(contourWithData)


    for contourWithData in allContoursWithData:
        if contourWithData.checkIfContourIsValid():
            validContoursWithData.append(contourWithData)


    validContoursWithData.sort(key = operator.attrgetter("intRectX"))
    strFinalString = ""
    rectList = []

    for contourWithData in validContoursWithData:
        cv2.rectangle(imgTestingNumbers,
                      (contourWithData.intRectX, contourWithData.intRectY),
                      (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
                      (0, 255, 0),
                      2)
        rectList.append((contourWithData.intRectX, contourWithData.intRectY,
                        contourWithData.intRectX + contourWithData.intRectWidth,
                        contourWithData.intRectY + contourWithData.intRectHeight))

    averageArea, averageY = findAverages(rectList)

    for contourWithData in validContoursWithData:
        imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,
                           contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]
        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
        npaROIResized = np.float32(npaROIResized)
        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k=2)
        strCurrentChar = str(chr(int(npaResults[0][0])))
        if strCurrentChar == 'X':
            strCurrentChar = 'x'
        if strCurrentChar == 'd' or strCurrentChar == 's':
            strCurrentChar = '-'
        
        if contourWithData.intRectY + contourWithData.intRectHeight  < averageY:
            spacingChar = '^'
        # elif strFinalString != '' and strFinalString[-1] == 'x' and strCurrentChar.isdigit():
        #     strFinalString += '^'
        else:
            spacingChar = ''
        strFinalString += spacingChar + strCurrentChar

    # print("\n" + strFinalString + "\n")

    # for testing purposes
    # cv2.imshow("imgTestingNumbers", imgTestingNumbers)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return strFinalString

######################################################################


# colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


# gives popup to edit what the scanner thought that the input was
def showInfo(s):
    # the following three lines suppress the extra tkinter box that would
    # show up with my dialog box
    # they are taken from
    # (next two lines are URL to keep under 80 chars)
    # http://stackoverflow.com/questions/17280637/
    # tkinter-messagebox-without-window
    # by user BuvinJ
    root = tk.Tk()
    root.overrideredirect(1)
    root.withdraw()
    response = simpledialog.askstring('Edit',
                                      'Make any changes you need:',
                                      initialvalue=s)
    return response


def findAverages(a):
    averageY = []
    averageArea = []
    for (x0, y0, x1, y1) in a:
        averageY.append((y0 + y1) / 2)
        averageArea.append((x1 - x0) * (y1 - y0))
    return listAvg(averageArea), listAvg(averageY)


def listAvg(a):
    if a == []:
        return 0
    total = 0
    for n in a:
        total += n
    return total / len(a)


##############################
# evaluate
##############################

def sign(n):
    return 1 if n >= 0 else -1


# takes a value and removes all occurences of that value froma  list
def removeAllFromList(a, value):
    for i in range(len(a)):
        if a[i] == value:
            a.pop(i)
    return a


# technically returns a random key from a dictionary and what it maps to
# i use this as a helper function for a special case where a dictionary
# only has one entry, so it returns that key and its value
def getOnlyElement(d):
    a = list(d.keys())
    key = a[0]
    return key, d[key]


# returns which to values of the domain return values of the codomain with
# opposite values
def oppositeSign(p, a, b, c):
    fA = evaluatePolynomial(p, a)
    fC = evaluatePolynomial(p, c)
    if sign(fA) == sign(fC):
        return b, c
    else:
        return a, c


# finds a root of a given function given a root between two values of the domain
def findRoot(p, a, b):
    epsilon = 10 ** -8
    if sign(evaluatePolynomial(p, a)) == sign(evaluatePolynomial(p, b)):
        return None
    elif abs(evaluatePolynomial(p, a)) < epsilon:
        return a
    elif abs(evaluatePolynomial(p, b)) < epsilon:
        return b
    elif abs(a - b) < epsilon:
        return abs(a + b) / 2
    else:
        fA = evaluatePolynomial(p, a)
        fB = evaluatePolynomial(p, b)
        c = (a * fB - b * fA) / (fB - fA)
        a, b = oppositeSign(p, a, b, c)
        return findRoot(p, a, b)


def subtractPolynomials(p3, p4):
    p1 = copy.copy(p3)
    p2 = copy.copy(p4)
    for key in p2:
        if key in p1:
            p1[key] = p1[key] - p2[key]
        else:
            p1[key] = -1 * p2[key]
    return p1


def findIntersection(p1, p2):
    p = subtractPolynomials(p1, p2)
    return findZeros(p)


# finds the intersections of all polynomials in a list a
def findAllIntersections(a):
    intersections = []
    for i in range(len(a) - 1):
        for j in range(i+ 1, len(a)):
            intersections.append(findIntersection(a[i], a[j]))                
    c =  flatten(intersections)
    return removeAllFromList(c, None)



# returns a dictionary with the degrees of a polynomial mapping to their
# coefficients given a polynomial entered as a string
# (of the form a_n x^n + ... + a_1 x + a_0)
def parsePolynomial(p):
    p = p.replace('-', '+-')
    d = dict()
    for term in p.split('+'):
        try:
            num = (float(term))
            if 0 not in d:
                d[0] = 0
            d[0] += num
        except:
            if term[-1] in string.ascii_lowercase:
                if 1 not in d:
                    d[1] = 0
                if term[:-1] == '':
                    coeff = 1
                elif term[:-1] == '-':
                    coeff = -1
                else:
                    coeff = float(term[:-1])
                d[1] += coeff
            else:
                coeff = getCoeff(term)
                exp = float(getExp(term))
                if exp not in d:
                    d[exp] = 0
                if coeff == '':
                    coeff = 1
                elif coeff == '-':
                    coeff = -1
                else:
                    coeff = float(coeff)
                d[exp] += coeff
    return d


# takes a parsed polynomial and gives back its original string
def parsedToString(p):
    result = ''
    first = True
    for key in sorted(p.keys())[::-1]:
        sgn = sign(p[key])
        signChar = '' if sgn == -1 or first else '+'
        if almostEqual(int(p[key]), p[key]):
            result += '%s%ix^%i' % (signChar, p[key], key)
            # result += '%s%fx^%f' % (signChar, p[key], key)
        elif True:
            result += '%s%0.2fx^%i' % (signChar, p[key], key)
        elif key == 1:
            try:
                result += '%s%ix' % (signChar, p[key])
            except:
                result += '%s%fx' % (signChar, p[key])
        else:
            result += str(p[key])
        first = False
    return result


def getExp(term):
    results = getCoeff(term[::-1])
    fixedResults = results[::-1]
    return fixedResults


def getCoeff(term):
    coeff = ''
    for c in term:
        if c == '-':
            coeff += c
        else:
            try:
                coeff += c
                float(coeff)
            except:
                return coeff[:-1]


def takeDerivative(polynomial):
    derivative = dict()
    for key in polynomial.keys():
        if key != 0:
            derivative[key - 1] = key * polynomial[key]
    return derivative


def indefiniteIntegral(polynomial):
    integral = dict()
    for key in polynomial.keys():
        integral[key + 1] = polynomial[key] / (key + 1)
    integral[0] = 'c'
    return integral


# returns value of a polynomial at x
def evaluatePolynomial(polynomial, x):
    total = 0
    for key in polynomial:
        total += x ** key * polynomial[key]
    return total


def derivativeAtPoint(polynomial, x):
    derivative = takeDerivative(polynomial)
    return evaluatePolynomial(derivative, x)


def nthDerivative(polynomial, n):
    for i in range(n):
        polynomial = takeDerivative(polynomial)
    return polynomial


def definiteIntegral(polynomial, a, b):
    integral = indefiniteIntegral(polynomial)
    del integral[0]
    fA = evaluatePolynomial(integral, a)
    fB = evaluatePolynomial(integral, b)
    return fB - fA + 0.0  # + 0.0 converts back to float


# finds all of the zeros of a given polynomial
def findZeros(polynomial):
    if len(polynomial) == 1:
        degree, value = getOnlyElement(polynomial)
        if degree == 0:
            if value != 0:
                return None
            else:  # this is the line y = 0
                return 'all reals'
        else:
            return [0]
    # all polynomials with two terms are easily solved, so we stop and solve
    elif len(polynomial) == 2:
        return solveLenTwo(polynomial)
    else:
        derivativeList = findZeros(takeDerivative(polynomial))
        derivativeList = removeAllFromList(derivativeList, None)
        if derivativeList == []:
            return findEdgeRoot(polynomial, 0, True, True)
        elif len(derivativeList) == 1:  # in this case we check both ends
            x = derivativeList[0]
            return findEdgeRoot(polynomial, x, True, True)
        else:
            roots = []
            derivativeList = sorted(derivativeList)
            xI = derivativeList[0]
            xN = derivativeList[-1]
            # start and end give the roots at the edges
            start = findEdgeRoot(polynomial, xI, True, False)
            end = findEdgeRoot(polynomial, xN, False, True)
            roots.append(start)
            roots.append(end)
            # check between each derivative - we can have at most one root
            # on this region
            for i in range(len(derivativeList) - 1):
                x1 = derivativeList[i]
                x2 = derivativeList[i + 1]
                roots.append(findRoot(polynomial, x1, x2))
                return removeAllFromList(flatten(roots), None)


def solveLenTwo(polynomial):
    roots = []
    smallerDegree = min(polynomial.keys())
    largerDegree = max(polynomial.keys())
    if smallerDegree != 0:  # like checking if we can factor out an x
        roots.append(0)
    minCoeff = polynomial[smallerDegree]
    maxCoeff = polynomial[largerDegree]
    rhs = - minCoeff / maxCoeff  # solving the equation for x**n
    # avoiding complex roots for even powers, if they are real then we have 
    # plus or minus the root
    if (largerDegree - smallerDegree) % 2 == 0 and rhs >= 0:
        root = rhs ** (1 / (largerDegree - smallerDegree))
        roots.extend([root , - root])
    elif (largerDegree - smallerDegree) % 2 == 1:
        roots.append(rhs ** (1 / (largerDegree - smallerDegree)))
    return roots


# finds the roots of a function when we do not have two derivatives to check
# between
# left and right are booleans and determine whether or not we check in that
# direction
def findEdgeRoot(polynomial, x, left, right):
    roots = []
    # we should only check for roots if either the second derivative is > 0 and
    # the function is less than zero or vice versa
    secondDerivative = nthDerivative(polynomial, 2)
    concavitySign = sign(evaluatePolynomial(secondDerivative, x))
    polySign = sign(evaluatePolynomial(polynomial, x))
    delta = -1  # the magnitude of delta is arbitrary, only the sign matters
    while left and concavitySign != polySign:
        if findRoot(polynomial, x, x + delta) != None:
            roots.append(findRoot(polynomial, x, x + delta))
            break
        else:
            delta -= 1
    delta = 1
    while right and concavitySign != polySign:
        if findRoot(polynomial, x, x + delta) != None:
            roots.append(findRoot(polynomial, x, x + delta))
            break
        else:
            delta += 1
    return roots


# flattens lists containing lists to a single list with only their elements
# we were shown a function in 112 lab that serves this same purpose,
# but were never given a chance to write it down
def flatten(a):
    if a == []:
        return a
    elif isinstance(a, list):
        return flatten(a[0]) + flatten(a[1:])
    else:
        return [a]


# checks if a string has multiple letters in it
def doubleLetters(s):
    for c in s:
        if c in string.ascii_lowercase and c != 'x':
            return True
    return False


# returns what to do with our results from scanning
def evaluateResults(data, s):
    try:
        temp = parsePolynomial(s)
        if 'x' not in s or doubleLetters(s):
            a = 1/0
        data.results = s
        data.polynomialScreen = True
        data.zeros.append(findZeros(parsePolynomial(s)))
        data.zeros = flatten(data.zeros)
        data.previousScreen = 'polynomial'
        if data.batchCounter != 3:
            data.functionList.append(data.results)
        polyPlot(data)
        return
    except:
        pass
    try:
        temp = evaluatePEMDAS(s)
        data.results = s
        data.results = data.results.replace('x', '*')
        data.arithmeticScreen = True
        data.previousScreen = 'arithmetic'
        return
    except:
        pass
    data.results = s
    data.noneFound = True


# this function takes in the index of an operation and a string and returns the
# numbers that are on either side of that operation, as well as the indices
# of the string corresponding to the first digit of the first number and
# the last digit of the last number
def findAdjacentNumbers(s, i):
    start = i - 1
    while((s[start].isdigit() or
           s[start] == '.') and start >= 0):
        start -= 1
    startNumber = float(s[start + 1: i])
    end = i + 1
    while(end < len(s) and
          (s[end].isdigit() or s[end] == '.')):
        end += 1
    endNumber = float(s[i + 1: end])
    return startNumber, endNumber, start + 1, end


def evaluate(a, b, op):
    # I just decided to store my strings to evaluate with '^' instead
    # of '**' just to that there is one less character which makes some of
    # my other code more simple
    if op == '^':
        return a ** b
    elif op == '*':
        return a * b
    elif op == '-':
        return a - b
    elif op == '+':
        return a + b
    elif op == '/':
        if b == 0:
            return 'error'
        else:
            return a / b


# this function finds the operation that we must do next, going through PEMDAS
def findOperation(s):
    i = s.find('^')
    if i != -1:
        return i
    i = s.find('*')
    j = s.find('/')
    # if only one of the two is present, we return that one
    # otherwise we provide which one occurs first, provided one of them occurs
    if i == -1 and j != -1:
        return j
    elif j == -1 and i != -1:
        return i
    elif i != -1 and j != -1:
        return i if i < j else j
    i = s.find('+')
    j = s.find('-')
    if i == -1 and j != -1:
        return j
    elif j == -1 and i != -1:
        return i
    else:
        return i if i < j else j


def evaluatePEMDAS(expression, l=None):
    if l == []:
        l.append(expression)
    s = ''
    for c in expression:
        if c == 'x':
            s += '*'
        else:
            s += c
    if s == '':
        return 0 if l is not None else 0, [0]
    # we will see if s can be written is a float
    # and thus also a positive or negative integer - in this case, we are done
    try:
        float(s)
        flt = True
    except:
        flt = False
    if flt:
        return float(s) if l is not None else float(s), [float(s)]
    else:
        i = findOperation(s)
        a, b, start, end = findAdjacentNumbers(s, i)
        newVal = evaluate(a, b, s[i])
        newStr = s.replace(s[start: end], str(newVal), 1)
        if isinstance(l, list):
            l.append(newStr)
        if l is None:
            return evaluatePEMDAS(newStr)
        else:
            return evaluatePEMDAS(newStr, l), l


# this function gets the slope and intercept of both sides of a
# linear equation
def getEquations(s):
    i = s.find('=')
    lhs = s[:i]
    rhs = s[i + 1:]
    return getSlopeIntercept(lhs), getSlopeIntercept(rhs)


def solveQuadratic(s):
    a, b, c = getCoefficients(s)
    roots = np.roots([a, b, c])
    return roots


def digitSearchBack(s, i):
    while i > 0 and not s[i].isdigit:
        i -= 1
    return i


def fixFunction(fn):
    new = ''
    for c in fn:
        if new != '' and c == 'x' and new[-1].isdigit():
            new += '*x'
        else:
            new += c
    new = new.replace('^', '**')
    return new


def polyPlot(data):
    pylab.figure()
    data.parsedList = [parsePolynomial(p) for p in data.functionList]
    data.intersections = findAllIntersections(data.parsedList.copy())
    data.intersections = [float('%0.2f' % n) for n in data.intersections]
    for fn in data.functionList:
        if fn is not None:
            p = fixFunction(fn)
            x = np.arange(-7, 7, .01)
            y = eval(p)
            pylab.plot(x, y, label=p)
    for zero in data.zeros:
        pylab.plot(zero, 0, 'gs')
    if len(data.parsedList) > 1:
        p = data.parsedList[1]
        points = [(xx, evaluatePolynomial(p, xx)) for xx in data.intersections]
        for (x1, y1) in points:
            pylab.plot(x1, y1, 'rs')
    pylab.legend(loc='upper left')
    pylab.savefig('resultGraph.png', dpi=500)


####################################################################
# graphics
####################################################################


# http://stackoverflow.com/questions/19306211/opencv-cv2-image-to-pygame-image
# function given by user High schooler with modification suggested by lgonato
def cvimage_to_pygame(image):
    """Convert cvimage into a pygame image"""
    return pygame.image.frombuffer(image.tostring(), image.shape[1::-1], "RGB")


# this function gives the x and y value needed to center a given block of
# text at a given x and y
def centerText(font, text, x, y):
    drawX = x - (font.size(text)[0] // 2)
    drawY = y - (font.size(text)[1] // 2)
    return (drawX, drawY)


# useful for finding if user clicked on text on screen
# returns cords of box surrounding the text
def giveTextBounds(font, text, x, y):
    x0 = x - (font.size(text)[0] // 2)
    y0 = y - (font.size(text)[1] // 2)
    x1 = x + (font.size(text)[0] // 2)
    y1 = y + (font.size(text)[1] // 2)
    return (x0, y0, x1, y1)


# once we apply this to what we want to write on the screen,
# we simply do surface.blit(label, cords)
def giveLabelCords(message, fontSize, cx, cy, color=BLACK):
    font = pygame.font.SysFont('cambriacambriamath', fontSize)
    label = font.render(message, True, color)
    cords = centerText(font, message, cx, cy)
    return label, cords


# moves the scanning rectangle on screen
def moveRect(data):
    (x, y) = data.webCords
    delta = 20
    x0, y0 = data.rectX + data.rectXoffset, data.rectY + data.rectYoffset
    x1, y1 = x0 + data.rectWidth, y0 + data.rectHeight
    # decides which edge we are moving
    if abs(x - x0) < delta and y < y1 and y > y0:
        data.leftEdge = True
    elif abs(x - x1) < delta and y < y1 and y > y0:
        data.rightEdge = True
    elif abs(y - y0) < delta and x < x1 and x > x0:
        data.lowerEdge = True
    elif abs(y - y1) < delta and x < x1 and x > x0:
        data.upperEdge = True


# this prevents my program from crashing if the users resizes the rectangle
# to be too small
def fixDimensions(data):
    if data.rectWidth < 20:
        data.rectWidth = 25
    elif data.rectHeight < 20:
        data.rectHeight = 25


def keyPressedWebcam(event, data):
    if event == ord('c'):
        if data.camSelected == 'batch' and data.batchCounter < 3:
            try:
                x, y = data.rectX + data.rectXoffset, data.rectY + data.rectYoffset
                width, height = data.rectWidth, data.rectHeight
                # method of indexing np array given at url by user martin giesler
                # http://stackoverflow.com/questions/903853/how-do-you-extract-a-column-from-a-multi-dimensional-array/903867
                rowIdx = np.array([i for i in range(y, y + height)])
                colIdx = np.array([j for j in range(x, x + width)])
                scanFrame = np.fliplr(data.frame[rowIdx[:, None], colIdx])
                cv2.imwrite('test123.png', scanFrame)
                data.results = main('test123.png')
                a = parsePolynomial(data.results)
                data.functionList.append(data.results)
                data.batchCounter += 1
                data.parsedList = [parsePolynomial(p) for p in data.functionList]
                data.intersections = findAllIntersections(data.parsedList.copy())
            except:
                pass
        else:
            data.arithmeticScreen = False
            data.polynomialScreen = False
            # resizing what we scan by indexing into np.array
            x, y = data.rectX + data.rectXoffset, data.rectY + data.rectYoffset
            width, height = data.rectWidth, data.rectHeight
            # same citation as above indexing into np arrays
            rowIdx = np.array([i for i in range(y, y + height)])
            colIdx = np.array([j for j in range(x, x + width)])
            scanFrame = np.fliplr(data.frame[rowIdx[:, None], colIdx])
            cv2.imwrite('test123.png', scanFrame)
            if data.batchCounter != 3:
                data.results = main('test123.png')
            data.webcam = False
            data.capturing = False
            evaluateResults(data, data.results)
            data.previousScreen = 'webcam'
    elif event == 275:  # ords for arrow keys
        data.rectXoffset += 10
    elif event == 276:
        data.rectXoffset -= 10
    elif event == 274:
        data.rectYoffset += 10
    elif event == 273:
        data.rectYoffset -= 10
    elif event == ord('q') or event == 8:  # 8 is backspace
        data.start = True
        data.webcam = False
        data.capturing = False


def keyPressedHelp(event, data):
    if event == K_LEFT and not data.cameraHelp and not data.graphHelp:
        data.help = False
        data.start = True
    elif event == K_LEFT and data.cameraHelp:
        data.cameraHelp = False
    elif event == K_LEFT and data.graphHelp:
        data.graphHelp = False
        data.cameraHelp = True
    elif event == K_RIGHT and not data.graphHelp and not data.cameraHelp:
        data.cameraHelp = True
    elif event == K_RIGHT and not data.graphHelp and data.cameraHelp:
        data.cameraHelp = False
        data.graphHelp = True
    elif event == K_RIGHT and data.graphHelp:
        data.help = False
        data.graphHelp = False
        data.start = True


def keyPressedPolynomial(event, data):
    if event == ord('z') and data.undoList != []:
        a = data.undoList.pop()
        data.functionList.append(a)
        data.deleted = True
        polyPlot(data)
    elif event == ord('y') and len(data.functionList) > 1 and data.undoList != []:
        a = data.functionList.pop()
        data.undoList.append(a)
        data.deleted = True
        polyPlot(data)


def keyPressedEdit(event, data, textinput):
    # evaluating our results
    if event == K_RETURN:
        data.changed = True
        data.functionList = []
        data.zeros = []
        data.intersections = []
        data.arithmeticScreen = False
        data.polynomialScreen = False
        data.results = textinput.get_text()
        data.editScreen = False
        textinput.erase_text()
        data.writtenResults = False
        evaluateResults(data, data.results)
        # data.resultsScreen = True


def keyPressed(event, data):
    if data.webcam:
        keyPressedWebcam(event, data)
    elif data.help:
        keyPressedHelp(event, data)
    elif data.polynomialScreen:
        keyPressedPolynomial(event, data)
    elif data.editScreen:
        keyPressedEdit(event, data)


def startMousePressed(data, x, y):
    (helpX0, helpY0, helpX1, helpY1) = data.helpBox
    (scanX0, scanY0, scanX1, scanY1) = data.scanBox
    if(x > helpX0 and x < helpX1 and y > helpY0 and y < helpY1):
        data.start = False
        data.help = True
    elif(x > scanX0 and x < scanX1 and y > scanY0 and y < scanY1):
        data.start = False
        data.intro = False
        data.webcam = True
        data.capturing = True


def arithmeticMousePressed(data, x, y):
    (backX0, backY0, backX1, backY1) = data.backBox
    (editX0, editY0, editX1, editY1) = data.editBox
    if(x > backX0 and x < backX1 and y > backY0 and y < backY1):
        data.confirmed = False
        data.webcam = True
        data.capturing = True
        data.results = None
        data.evaluatedResults = None
        data.clearPlot = True
    elif(x > editX0 and x < editX1 and y > editY0 and y < editY1):
        data.editScreen = True
        data.previousScreen = 'arithmetic'
        data.resultsSreen = False


def polynomialMousePressed(data, x, y):
    (backX0, backY0, backX1, backY1) = data.backBox
    (editX0, editY0, editX1, editY1) = data.editBox
    if(x > backX0 and x < backX1 and y > backY0 and y < backY1):
        data.confirmed = False
        data.webcam = True
        data.capturing = True
        data.results = None
        data.evaluatedResults = None
        data.drawPolynomialScreen = False
        data.clearPlot = True
        data.zeros = []
        data.functionList = []
        data.batchCounter = 0
        data.camSelected = 'single'
    elif(x > editX0 and x < editX1 and y > editY0 and y < editY1):
        data.polynomialScreen = False
        data.editScreen = True
        data.previousScreen = 'polynomial'


def noneFoundMousePressed(data, x, y):
    (backX0, backY0, backX1, backY1) = data.backBox
    (editX0, editY0, editX1, editY1) = data.editBox
    if(x > backX0 and x < backX1 and y > backY0 and y < backY1):
        data.confirmed = False
        data.noneFound = False
        data.webcam = True
        data.capturing = True
        data.results = None
        data.evaluatedResults = None
        data.clearPlot = True
    elif(x > editX0 and x < editX1 and y > editY0 and y < editY1):
        data.noneFound = False
        data.editScreen = True
        data.previousScreen = 'none'


def mousePressed(event, data):
    (x, y) = pygame.mouse.get_pos()
    if data.start:
        startMousePressed(data, x, y)
    elif data.arithmeticScreen:
        arithmeticMousePressed(data, x, y)
    elif data.polynomialScreen:
        polynomialMousePressed(data, x, y)
    elif data.noneFound:
        noneFoundMousePressed(data, x, y)


def init(data):
    data.zeros = []
    data.clearPlot = True
    data.capturing = False
    data.frame = None
    data.webcam = False
    data.help = False
    data.events = None
    data.start = True
    data.helpBox = None
    data.scanBox = None
    data.backBox = None
    data.derivativeBox = None
    data.integralBox = None
    data.zerosBox = None
    data.eval = None
    data.results = None
    data.resultsScreen = False
    data.plotted = False
    data.arithmeticScreen = False
    data.polynomialScreen = False
    data.noneFound = False
    data.editBox = None
    data.editScreen = False
    data.result = True
    data.confirmed = False
    data.evaluatedResults = None
    data.functionList = []
    data.parsedList = []
    data.writtenResults = False
    data.rectX = int(.2 * data.width)
    data.rectY = int(.2 * data.height)
    data.rectWidth = int(.6 * data.width)
    data.rectHeight = int(.3 * data.height)
    data.rectXoffset = 0
    data.rectYoffset = 0
    data.rectSizeMultiplier = 1
    data.scanColor = BLACK
    data.helpColor = BLACK
    data.rectDragging = False
    data.webCords = (-1, -1)
    data.leftEdge = False
    data.rightEdge = False
    data.upperEdge = False
    data.lowerEdge = False  # lower y value, higher on screen
    data.batchBox = None
    data.singleBox = None
    data.doneBox = None
    data.backColor = WHITE
    data.singleColor = WHITE
    data.batchColor = WHITE
    data.loadColor = WHITE
    data.camSelected = 'single'
    data.batchCounter = 0
    data.shutterCenter = None
    data.shutterInnerRadius = 35
    data.shutterColor = BLACK
    data.previousScreen = None
    data.popupClicked = False
    data.polynomialMessage = 'Click on functions to access more options!'
    data.deleted = False
    data.graphHelp = False
    data.cameraHelp = False
    data.undoList = []  # framework for undo redo taken from 15-112 events page
    data.intersections = []
    data.editColor = BLACK
    data.integralColor = BLACK
    data.derivativeColor = BLACK
    data.zerosColor = BLACK
    data.deleteColor = BLACK
    data.changed = False
    data.parsedList = []


def drawEditScreen(screen, data, textinput, clock):
    subFont = pygame.font.SysFont('cambriacambriamath', 30)
    pygame.display.set_caption('Polynomial Time')
    screen.fill(WHITE)
    events = pygame.event.get()
    if textinput.get_text() == '' and not data.writtenResults:
        textinput.add_text(data.results)
        data.writtenResults = True
    for event in events:
        if event.type == pygame.QUIT:
            exit()
        elif event.type == pygame.KEYDOWN:
            keyPressedEdit(event.key, data, textinput)
    textinput.update(events)
    editText, editFont = giveLabelCords('Edit:', 40, .3 * data.width,
                                        .4 * data.height)
    backMessage, backFont = giveLabelCords('Back', 30,
                                            .1 * data.width, .05 * data.height,
                                            data.backColor)
    data.backBox = giveTextBounds(subFont, 'Back', 
                                  .1 * data.width, .05 * data.height)
    screen.blit(textinput.get_surface(), (.4 * data.width // 2, data.height // 2))
    screen.blit(editText, editFont)
    (x, y) = pygame.mouse.get_pos()
    if(x > data.backBox[0] and x < data.backBox[2] and
        y > data.backBox[1] and y < data.batchBox[3]):
        data.backColor = BLUE
        if pygame.mouse.get_pressed() == (1, 0, 0):
            data.editScreen = False
            if data.previousScreen == 'polynomial':
                data.polynomial = True
            elif data.previousScreen == 'arithmetic':
                data.arithmetic = True
            elif data.previousScreen == 'webcam':
                data.webcam = True
            elif data.previousScreen == 'none':
                data.noneFound = True
    else:
        data.backColor = BLACK
    screen.blit(backMessage, backFont)
    pygame.display.update()
    clock.tick(30)


def drawWebcam(screen, data):
    pygame.display.set_caption('Polynomial Time')
    cap = cv2.VideoCapture(0)
    while data.webcam:
        if data.capturing:
            ret, frame = cap.read()
        frame = np.fliplr(frame)
        data.frame = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        myImg = cvimage_to_pygame(frame)
        events = pygame.event.get()
        for event in events:
            if event.type == QUIT:
                cap.release()
                data.capturing = False
                cv2.destroyAllWindows()
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                keyPressed(event.key, data)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mousePressed(event, data)
                (x, y) = pygame.mouse.get_pos()
                (x1, y1) = data.shutterCenter
                if distance(x, y, x1, y1) < data.shutterInnerRadius:
                    keyPressedWebcam(ord('c'), data)
                data.rectDragging = True
            elif event.type == MOUSEBUTTONUP:
                data.rectDragging = False
                data.leftEdge = False
                data.rightEdge = False
                data.lowerEdge = False
                data.upperEdge = False
            elif event.type == MOUSEMOTION and data.rectDragging:
                data.webCords = pygame.mouse.get_pos()
                moveRect(data)
                (x, y) = data.webCords
                adjustRectangle(data, x, y)
        if data.capturing:
            screen.blit(myImg, (0, 0))
        if data.result and data.confirmed:
            data.webcam = False
            data.capturing = False
            data.resultsScreen = True
        drawWebcamTransparency(screen, data)
        pygame.display.flip()
        pygame.display.update()


def adjustRectangle(data, x, y):
    if data.leftEdge and data.rectWidth > 20:
        data.rectWidth -= x - data.rectX
        data.rectX = x
    elif data.rightEdge and data.rectWidth > 20:
        data.rectWidth = x - data.rectX
    elif data.lowerEdge and data.rectHeight > 20:
        data.rectHeight -= y - data.rectY
        data.rectY = y
    elif data.upperEdge and data.rectHeight > 20:
        data.rectHeight = y - data.rectY
    fixDimensions(data)


def drawWebcamTransparency(screen, data):
    # transparancy help given by user Sloth from link:
    # http://stackoverflow.com/questions/17581545/drawn-surface-transparency-in-pygame
    t = pygame.Surface((data.width, data.height))
    t.fill(RED) # erases this color, doesn't really matter which
    t.set_colorkey(RED)
    pygame.draw.circle(t, data.shutterColor, (data.width // 2, int(.85 * data.height)), 35)
    pygame.draw.circle(t, BLACK, (data.width // 2, int(.85 * data.height)), 42, 5)
    data.shutterCenter = (data.width // 2, int(.85 * data.height))
    pygame.draw.rect(t, BLACK, (data.rectX + data.rectXoffset,
                     data.rectY + data.rectYoffset,
                     data.rectWidth, data.rectHeight), 5)
    pygame.draw.rect(t, BLACK, (0, 0, data.width, .15 * data.height))
    t.set_alpha(128)
    screen.blit(t, (0, 0))
    addMenuText(screen, data)


def addMenuText(screen, data):
    subFont = pygame.font.SysFont('cambriacambriamath', 30)
    backMessage, backFont = giveLabelCords('Back', 30,
                                            .1 * data.width, .1 * data.height, 
                                            data.backColor)
    batchMessage, batchFont = giveLabelCords('Batch', 30,
                                             .63 * data.width, .1 * data.height, 
                                             data.batchColor)
    singleMessage, singleFont = giveLabelCords('Single', 30, .37 * data.width,
                                                .1 * data.height,
                                                data.singleColor)
    doneMessage, doneFont = giveLabelCords('Done (%i)' % data.batchCounter, 30,
                                            .85 * data.width,
                                           .1 * data.height, data.loadColor)
    data.backBox = giveTextBounds(subFont, 'Back', .1 * data.width,
                                  .1 * data.height)
    data.batchBox = giveTextBounds(subFont, 'Batch', .63 * data.width,
                                   .1 * data.height)
    data.singleBox = giveTextBounds(subFont, 'Single', .37 * data.width,
                                    .1 * data.height)
    data.doneBox = giveTextBounds(subFont, 'Done (%i)' % data.batchCounter,
                                  .85 * data.width, .1 * data.height)
    screen.blit(backMessage, backFont)
    screen.blit(batchMessage, batchFont)
    screen.blit(singleMessage, singleFont)
    if data.batchCounter > 0:
        screen.blit(doneMessage, doneFont)
    (x, y) = pygame.mouse.get_pos()
    if(x > data.backBox[0] and x < data.backBox[2]
        and y > data.backBox[1] and y < data.backBox[3]):
        data.backColor = BLUE
        if pygame.mouse.get_pressed() == (1, 0, 0):
            data.start = True
            data.webcam = False
            data.capturing = False
    else:
        data.backColor = WHITE
    if(x > data.batchBox[0] and x < data.batchBox[2]
        and y > data.batchBox[1] and y < data.batchBox[3]
         or data.camSelected == 'batch'):
        data.batchColor = BLUE
        if pygame.mouse.get_pressed() == (1, 0, 0):  # left mouse click
            data.camSelected = 'batch'
    else:
        data.batchColor = WHITE
    if(x > data.singleBox[0] and x < data.singleBox[2]
        and y > data.singleBox[1] and y < data.singleBox[3] 
        or data.camSelected == 'single'):
        data.singleColor = BLUE
        if pygame.mouse.get_pressed() == (1, 0, 0) and data.batchCounter == 0:
            data.camSelected = 'single'
    else:
        data.singleColor = WHITE
    if(x > data.doneBox[0] and x < data.doneBox[2]
        and y > data.doneBox[1] and y < data.doneBox[3]):
        data.loadColor = BLUE
        if pygame.mouse.get_pressed() == (1, 0, 0):
            data.batchCounter = 3
            keyPressedWebcam(ord('c'), data)
    else:
        data.loadColor = WHITE
    if(distance(x, y, data.shutterCenter[0], data.shutterCenter[1]) < data.shutterInnerRadius):
        data.shutterColor = WHITE
    else:
        data.shutterColor = BLACK


def createFileDropdown(screen, data):
    points = ((int(.8 * data.width), int(.15 * data.height)),
             (int(.82 * data.width), int(.17 * data.height)),
                (int(.78 * data.width), int(.17 * data.height)))
    pygame.draw.polygon(screen, BLACK, points)
    pygame.draw.rect(screen, BLACK, ((int(.65 * data.width),
                                      int(.17 * data.height)),
                                     (int(.3 * data.width),
                                      int(.3 * data.height))))


def drawStartScreen(screen, data):
    events = pygame.event.get()
    pygame.display.set_caption('Polynomial Time')
    screen.fill(WHITE)
    pygame.init()
    background = cv2.imread('homeBG.jpg')
    background = cvimage_to_pygame(background)
    background = pygame.transform.scale(background,
                                        (int(750 * .9), int(.9 * 1000)))
    screen.blit(background, (0, 0))
    addStartScreenText(screen, data)
    pygame.draw.line(screen, WHITE, (data.width // 2 - 135, 175),
                     (data.width // 2 + 135, 175))
    (x, y) = pygame.mouse.get_pos()
    if(x > data.scanBox[0] and x < data.scanBox[2]
        and y > data.scanBox[1] and y < data.scanBox[3]):
        data.scanColor = BLUE
    else:
        data.scanColor = WHITE
    if(x > data.helpBox[0] and x < data.helpBox[2]
        and y > data.helpBox[1] and y < data.helpBox[3]):
        data.helpColor = BLUE
    else:
        data.helpColor = WHITE
    for event in events:
        if event.type == QUIT:
            cv2.destroyAllWindows()
            pygame.quit()
            sys.exit()
    pygame.display.update()


def addStartScreenText(screen, data):
    subFont = pygame.font.SysFont('cambriacambriamath', 30)
    data.scanBox = giveTextBounds(subFont, 'Scan',
                                  data.width // 2, data.height // 2)
    data.helpBox = giveTextBounds(subFont, 'Help',
                                  data.width // 2, .66 * data.height)
    headerBox = giveTextBounds(subFont, 'Polynomial Time',
                                data.width // 2, .3 * data.height)
    t = pygame.Surface((data.width, data.height))
    t.fill(RED)
    t.set_colorkey(RED)
    pygame.draw.rect(t, (147,112,219), (data.width // 2 - 150, 100,
                                300, 300))
    t.set_alpha(200)
    screen.blit(t, (0, 0))
    headerMessage, headerFont = giveLabelCords('Polynomial Time', 33,
                                                data.width // 2,
                                                .3 * data.height,
                                                WHITE)
    scanMessage, scanFont = giveLabelCords('Scan', 30, data.width // 2,
                                           data.height // 2,
                                           data.scanColor)
    helpMessage, helpFont = giveLabelCords('Help', 30, data.width // 2,
                                           .66 * data.height,
                                           data.helpColor)
    screen.blit(headerMessage, headerFont)
    screen.blit(scanMessage, scanFont)
    screen.blit(helpMessage, helpFont)


def drawHelpScreen(screen, data):
    events = pygame.event.get()
    screen.fill(WHITE)
    pygame.init()
    background = cv2.imread('introHelp.png')
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    background = cvimage_to_pygame(background)
    background = pygame.transform.scale(background,
                                        (data.width + 5, data.height + 5))
    screen.blit(background, (0, 0))
    if data.cameraHelp:
        background = cv2.imread('cameraScreen.png')
        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        background = cvimage_to_pygame(background)
        background = pygame.transform.scale(background, (data.width, data.height))
        screen.blit(background, (0, 0))
    elif data.graphHelp:
        background = cv2.imread('graphScreen.png')
        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        background = cvimage_to_pygame(background)
        background = pygame.transform.scale(background,
                                            (data.width + 5, data.height + 5))
        screen.blit(background, (0, 0))
        pygame.draw.rect(screen, WHITE, (0, 0, data.width, 8))
    t = pygame.Surface((data.width, data.height))
    t.fill(RED)
    t.set_colorkey(RED)
    pygame.draw.rect(t, BLACK, (0, .85 * data.height, data.width,
                        .15 * data.height))
    t.set_alpha(145)
    screen.blit(t, (0, 0))
    helpMessage, helpFont = giveLabelCords('Move left or right with arrow keys!', 35,
                                           data.width // 2, .92 * data.height,
                                           WHITE)
    screen.blit(helpMessage, helpFont)
    for event in events:
        if event.type == QUIT:
            cv2.destroyAllWindows()
            pygame.quit()
            sys.exit()
    pygame.display.update()


def drawNoneFoundScreen(screen, data):
    events = pygame.event.get()
    screen.fill(WHITE)
    subFont = pygame.font.SysFont('cambriacambriamath', 30)
    pygame.init()
    backMessage, backFont = giveLabelCords('Back', 30,
                                            .1 * data.width, .05 * data.height,
                                            data.backColor)
    data.backBox = giveTextBounds(subFont, 'Back', 
                                  .1 * data.width, .05 * data.height)
    editMessage, editFont = giveLabelCords('Edit', 30, .9 * data.width,
                                          .05 * data.height, data.editColor)
    data.editBox = giveTextBounds(subFont, 'Edit', 
                                  .9 * data.width, .05 * data.height)
    noneMessage, noneFont = giveLabelCords('None Found', 60,
                                                        data.width // 2,
                                                       .05 * data.height)
    tryAgainMessage1, tryAgainFont1 = giveLabelCords("Scanned '%s' and couldn't interpret" % data.results,
                                                    30, data.width // 2,
                                                    .45 * data.height)
    tryAgainMessage2, tryAgainFont2 = giveLabelCords("Try scanning again or editing",
                                                    30, data.width // 2,
                                                    .55 * data.height)
    screen.blit(backMessage, backFont)
    screen.blit(editMessage, editFont)
    screen.blit(noneMessage, noneFont)
    screen.blit(tryAgainMessage1, tryAgainFont1)
    screen.blit(tryAgainMessage2, tryAgainFont2)
    (x, y) = pygame.mouse.get_pos()
    (editX0, editY0, editX1, editY1) = data.editBox
    (backX0, backY0, backX1, backY1) = data.backBox
    if(x > editX0 and x < editX1 and y > editY0 and y < editY1):
        data.editColor = BLUE
    else:
        data.editColor = BLACK
    if(x > backX0 and x < backX1 and y > backY0 and y < backY1):
        data.backColor = BLUE
    else:
        data.backColor = BLACK
    for event in events:
        if event.type == QUIT:
            cv2.destroyAllWindows()
            pygame.quit()
            sys.exit()
    pygame.display.update()


def truncateLists(data):
    if not isinstance(data.intersections, list):
        data.interections = list(data.intersections)
    if not isinstance(data.zeros, list):
        data.zeros = list(data.zeros)
    for i in range(len(data.intersections)):
        a = truncate(data.intersections[i], 4)
        data.intersections[i] = a
    for i in range(len(data.zeros)):
        data.zeros[i] = truncate(data.zeros[i], 4)

# still have to fix image resizing - currently ruins text when made bigger
def drawPolynomialScreen(screen, data):
    truncateLists(data)
    events = pygame.event.get()
    screen.fill(WHITE)
    subFont = pygame.font.SysFont('cambriacambriamath', 30)
    pygame.init()
    myPlot = cv2.imread('resultGraph.png')
    myPlot = cvimage_to_pygame(myPlot)
    myPlot = pygame.transform.scale(myPlot, (int(.7 * data.width),
                                    int(.7 * data.height)))
    plotRect = myPlot.get_rect(center=screen.get_rect().center)
    screen.blit(myPlot, ((int(1.7 * plotRect[0]), int(1.2 * plotRect[1]))))
    polynomialSub, polynomialSubFont = giveLabelCords(data.polynomialMessage,
                                        30, .5 * data.width, .2 * data.height)
    backMessage, backFont = giveLabelCords('Back', 30,
                                            .1 * data.width, .05 * data.height,
                                            data.backColor)
    data.backBox = giveTextBounds(subFont, 'Back', 
                                  .1 * data.width, .05 * data.height)
    editMessage, editFont = giveLabelCords('Edit', 30, .9 * data.width,
                                          .05 * data.height, data.editColor)
    data.editBox = giveTextBounds(subFont, 'Edit', 
                                  .9 * data.width, .05 * data.height)
    polynomialMessage, polynomialFont = giveLabelCords('Polynomial', 60,
                                                        data.width // 2,
                                                       .05 * data.height)
    if len(data.functionList) == 1:
        zerosMessage, zerosFont = giveLabelCords('Zeros are: %s' % str(data.zeros),
                                                 30, data.width // 2, .9 * data.height)
        screen.blit(zerosMessage, zerosFont)
    else:
        intersectionMessage, intersectionFont = giveLabelCords('Intersection Points: %s' % str(data.intersections),
                                                                30, data.width // 2, .9 * data.height)
        screen.blit(intersectionMessage, intersectionFont)
    screen.blit(backMessage, backFont)
    screen.blit(editMessage, editFont)
    screen.blit(polynomialMessage, polynomialFont)
    screen.blit(polynomialSub, polynomialSubFont)
    drawFunctions(screen, data)
    (x, y) = pygame.mouse.get_pos()
    (editX0, editY0, editX1, editY1) = data.editBox
    (backX0, backY0, backX1, backY1) = data.backBox
    if(x > editX0 and x < editX1 and y > editY0 and y < editY1):
        data.editColor = BLUE
    else:
        data.editColor = BLACK
    if(x > backX0 and x < backX1 and y > backY0 and y < backY1):
        data.backColor = BLUE
    else:
        data.backColor = BLACK
    for event in events:
        if event.type == QUIT:
            cv2.destroyAllWindows()
            pygame.quit()
            sys.exit()
    pygame.display.update()


# draw function boxes later, and only activate in certain instances
def drawFunctions(screen, data):
    subFont = pygame.font.SysFont('cambriacambriamath', 40)
    yMargin = .4 * data.height
    for i in range(len(data.functionList)):
        fn = data.functionList[i]
        fnMessage, fnFont = giveLabelCords(fixFunction(fn), 25, .15 * data.width,
                                            yMargin)
        screen.blit(fnMessage, fnFont)
        fnBox = giveTextBounds(subFont, fn, .15 * data.width, yMargin)
        (x, y) = pygame.mouse.get_pos()
        # if pygame.mouse.get_pressed() == (1, 0, 0): print(42)
        if pygame.mouse.get_pressed() == (0, 0, 1):  # right click
            data.popupClicked = (False, i)
        if(x > fnBox[0] and x < fnBox[2] and y > fnBox[1] and y < fnBox[3]
            and pygame.mouse.get_pressed() == (1, 0, 0)
            or data.popupClicked  == (True, i)):
            x1 = (fnBox[0] + fnBox[2]) / 2
            y1 = (fnBox[1] + fnBox[3]) / 2
            showPopup(screen, data, x1, y1, fn)
            data.popupClicked = (True, i)
        yMargin += 50
        if data.deleted:
            data.deleted = False
            break


def showPopup(screen, data, x, y, fn):
    subFont = pygame.font.SysFont('cambriacambriamath', 30)
    (x0, y0, x1, y1) = (x, y, x + 125, int(.62 * y))
    t = pygame.Surface((data.width, data.height))
    t.fill(RED)
    t.set_colorkey(RED)
    integralMessage, integralFont = giveLabelCords('Show Integral', 20,
                                                    (2*x0 + x1)/2, 1.1 * y0,
                                                    data.integralColor)
    derivativeMessage, derivativeFont = giveLabelCords('Show Derivative', 20,
                                                    (2*x0 + x1)/2, 1.25 * y0,
                                                    data.derivativeColor)
    zerosMessage, zerosFont = giveLabelCords('Show Zeros', 20,
                                             (2*x0 + x1)/2, 1.4 * y0,
                                             data.zerosColor)
    deleteMessage, deleteFont = giveLabelCords('Delete', 20,
                                                (2*x0 + x1)/2, 1.55 * y0,
                                                data.deleteColor)
    integralBox = giveTextBounds(subFont, 'Show Integral',
                                (2*x0 + x1)/2, 1.1 * y0)
    derivativeBox = giveTextBounds(subFont, 'Show Derivative',
                                (2*x0 + x1)/2, 1.25 * y0)
    zerosBox = giveTextBounds(subFont, 'Show Zeros', (2*x0 + x1)/2, 1.4 * y0)
    deleteBox = giveTextBounds(subFont, 'Delete', (2*x0 + x1)/2, 1.55 * y0)
    xP, yP = pygame.mouse.get_pos()
    if(xP > integralBox[0] and xP < integralBox[2] and 
        yP > integralBox[1] and yP < integralBox[3]):
        if pygame.mouse.get_pressed() == (1, 0, 0) and len(data.functionList) <= 2:
            data.popupClicked = False
            integral = indefiniteIntegral(parsePolynomial(fn))
            del integral [0]
            integral = parsedToString(integral)
            data.functionList.append(integral)
            polyPlot(data)
            return
        else:
            data.integralColor = WHITE
    else:
        data.integralColor = BLACK
    if(xP > derivativeBox[0] and xP < derivativeBox[2] and 
        yP > derivativeBox[1] and yP < derivativeBox[3]):
        if pygame.mouse.get_pressed() == (1, 0, 0) and len(data.functionList) <= 2:
            data.popupClicked = False
            derivative = takeDerivative(parsePolynomial(fn))
            derivative = parsedToString(derivative)
            data.functionList.append(derivative)
            polyPlot(data)
            return
        else:
            data.derivativeColor = WHITE
    else:
        data.derivativeColor = BLACK
    if(xP > zerosBox[0] and xP < zerosBox[2] and 
        yP > zerosBox[1] and yP < zerosBox[3]):
        if pygame.mouse.get_pressed() == (1, 0, 0):
            zeros = findZeros(parsePolynomial(fn))
            for zero in zeros:
                data.zeros.append((zero, 0))
            polyPlot(data)
            return
        else:
            data.zerosColor = WHITE
    else:
        data.zerosColor = BLACK
    if(xP > deleteBox[0] and xP < deleteBox[2] and 
        yP > deleteBox[1] and yP < deleteBox[3]):
        if pygame.mouse.get_pressed() == (1, 0, 0):
            if len(data.functionList) <= 1:
                data.polynomialMessage = 'Need at least one function'
            else:
                data.functionList.remove(fn)
                data.undoList.append(fn)
                data.deleted = True
                polyPlot(data)
                return
        else:
            data.deleteColor = WHITE
    else:
        data.deleteColor = BLACK
    pygame.draw.rect(t, BLUE, (x0, y0, x1, y1))
    t.set_alpha(145)
    screen.blit(t, (0, 0))
    screen.blit(integralMessage, integralFont)
    screen.blit(derivativeMessage, derivativeFont)
    screen.blit(zerosMessage, zerosFont)
    screen.blit(deleteMessage, deleteFont)


def drawArithmeticScreen(screen, data):
    events = pygame.event.get()
    screen.fill(WHITE)
    subFont = pygame.font.SysFont('cambriacambriamath', 30)
    pygame.init()
    backMessage, backFont = giveLabelCords('Back', 30,
                                            .1 * data.width, .05 * data.height,
                                            data.backColor)
    data.backBox = giveTextBounds(subFont, 'Back', 
                                  .1 * data.width, .05 * data.height)
    editMessage, editFont = giveLabelCords('Edit', 30, .9 * data.width,
                                          .05 * data.height, data.editColor)
    data.editBox = giveTextBounds(subFont, 'Edit', 
                                  .9 * data.width, .05 * data.height)
    arithmeticMessage, arithmeticFont = giveLabelCords('Arithmetic', 60,
                                                        data.width // 2,
                                                       .05 * data.height)
    ans, l = evaluatePEMDAS(data.results, [])
    ansMessage, ansBox = giveLabelCords(str(ans), 30,
                                        data.width // 2, data.height * .6)
    startY = .65 * data.height
    for exp in l[::-1]:
        expMessage, expBox = giveLabelCords(str(exp), 30, data.width // 2,
                                            startY)
        screen.blit(expMessage, expBox)
        startY -= .1 * data.height
    screen.blit(backMessage, backFont)
    screen.blit(editMessage, editFont)
    screen.blit(arithmeticMessage, arithmeticFont)
    (x, y) = pygame.mouse.get_pos()
    (editX0, editY0, editX1, editY1) = data.editBox
    (backX0, backY0, backX1, backY1) = data.backBox
    if(x > editX0 and x < editX1 and y > editY0 and y < editY1):
        data.editColor = BLUE
    else:
        data.editColor = BLACK
    if(x > backX0 and x < backX1 and y > backY0 and y < backY1):
        data.backColor = BLUE
    else:
        data.backColor = BLACK
    for event in events:
        if event.type == QUIT:
            cv2.destroyAllWindows()
            pygame.quit()
            sys.exit()
    pygame.display.update()


def draw(data):
    screen = pygame.display.set_mode((data.width, data.height))
    events = pygame.event.get()
    for event in events:
            if event.type == QUIT:
                if data.capturing:
                    cap.release()
                data.capturing = False
                cv2.destroyAllWindows()
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                keyPressed(event.key, data)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mousePressed(event, data)
    if data.webcam:
        drawWebcam(screen, data)
    elif data.start:
        drawStartScreen(screen, data)
    elif data.help:
        drawHelpScreen(screen, data)
    elif data.arithmeticScreen:
        drawArithmeticScreen(screen, data)
    elif data.polynomialScreen:
        drawPolynomialScreen(screen, data)
    elif data.noneFound:
        drawNoneFoundScreen(screen, data)


# run function and data/event structure based off of eventsExample0.py
# from 15-112 course page
def run(width=640, height=480):
    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    init(data)
    textinput = pygame_textinput.TextInput()
    clock = pygame.time.Clock()
    while True:
        if not data.editScreen:
            draw(data)
            textinput = pygame_textinput.TextInput()
            clock = pygame.time.Clock()
        else:
            screen = pygame.display.set_mode((data.width, data.height))
            drawEditScreen(screen, data, textinput, clock)


# run()
