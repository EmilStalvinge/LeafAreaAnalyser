import cv2
import tkinter as tk
from future.moves.tkinter import filedialog

from pyimagesearch.transform import four_point_transform
from tkinter import *
from tkinter import messagebox
import numpy as np

# **************   variables  *******************************************

floodx = 0
floody = 0
pencilsize = 3
cord = [0, 0, 0, 0, 0, 0, 0, 0]
nummer = 0
a4area = 624                    # area of a4 papper 624cm^2

# *****************    picture input    ****************************************
root = tk.Tk()
width = root.winfo_screenwidth()
height = root.winfo_screenheight()
root.withdraw()

root.filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                           filetypes=(("jpeg files", "*.jpg"),
                                                      ("all files", "*.*")))

img = cv2.imread(root.filename)


# ****************       functions          ***************************************


def resizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)


def choose_points(event, x, y, flags, param):
    global cord, nummer
    if event == cv2.EVENT_LBUTTONDOWN:
        cord[nummer] = x
        nummer = nummer + 1
        cord[nummer] = y

    if event == cv2.EVENT_LBUTTONUP:
        nummer = nummer + 1
        if nummer == 8:
            cv2.destroyAllWindows()
            # messagebox.showinfo("Done", "the cordinates are:    " + repr(cord) + "   cm^2")
            cv2.setMouseCallback('points', lambda *args: None)


def greenfilter(img):
    lower_green = np.array([0, 0, 0])  ##[R value, G value, B value] #60
    upper_green = np.array([greenlow, 255, 255])  # 200
    # img = cv2.bitwise_not(img)
    return cv2.cvtColor(cv2.inRange(img, lower_green, upper_green), cv2.COLOR_GRAY2RGB)


def calcBlackpart(lastimg):
    gray = cv2.cvtColor(lastimg, cv2.COLOR_BGR2GRAY)  # compute all black pixels
    _, gray = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    # cv2.imshow('gray', gray)
    # cv2.waitKey(0)
    cv2.imwrite(root.filename + "analyzed.jpg", lastimg)
    cntnotblack = cv2.countNonZero(gray)  # get all non black Pixels
    # get pixel count of image
    height, width = gray.shape
    cntPixels = height * width
    cntBlackPart = (cntPixels - cntnotblack) / cntPixels  # cntWhitePart = cntNotBlack / cntPixels
    # print(cntBlackPart)
    Area = round(cntBlackPart * a4area * 10) / 10
    messagebox.showinfo("Done", "the black area is:    "
                        + repr(Area) + "   cm^2" "     picture saved")
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # root.destroy()
    sys.exit()


drawing = False  # true if mouse is pressed
pt1_x, pt1_y = None, None
left = False


# mouse callback function
def line_drawing(event, x, y, flags, param):
    global pt1_x, pt1_y, drawing, left

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        left = True
        pt1_x, pt1_y = x, y
    elif event == cv2.EVENT_RBUTTONDOWN:
        left = False
        drawing = True
        pt1_x, pt1_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing is True and left is True:
            cv2.line(result, (pt1_x, pt1_y), (x, y), color=(0, 0, 0), thickness=pencilsize + 2)

            pt1_x, pt1_y = x, y
        elif drawing is True and left is False:
            # cv2.line(img, (pt1_x, pt1_y), (x, y), color=(20, 100, 10), thickness=3)
            result[(y - (pencilsize + 1)):(y + pencilsize + 1), (x - (pencilsize + 1)):(x + pencilsize + 1)] = \
                org[(y - (pencilsize + 1)):(y + pencilsize + 1), (x - (pencilsize + 1)):(x + pencilsize + 1)]

            # print(str(x) + "     " + str(y))
            pt1_x, pt1_y = x, y
    elif event == cv2.EVENT_LBUTTONUP or cv2.EVENT_RBUTTONUP:
        drawing = False
        # cv2.line(result, (pt1_x, pt1_y), (x, y), color=(0, 0, 0), thickness=3)

    # if event == cv2.EVENT_LBUTTONDBLCLK:
    #    calcBlackpart(result)


def flood(event, x, y, flags, param):
    global floodx, floody
    if event == cv2.EVENT_LBUTTONDOWN:
        floodx = x
        floody = y


def nothing(x):
    pass


def floodreplace(imgg):
    width, height, channels = imgg.shape

    # print(height, width, channels)

    for x in range(0, width - 1):
        for y in range(0, height - 1):
            if imgg[x, y, 0] > 250:
                imgg[x, y] = org[x, y]

    return imgg


# ***************************      MAIN   *****************************************


# creates the window enable mouse click
img = resizeWithAspectRatio(img, height=int(height * 0.9))
cv2.namedWindow(root.filename)
cv2.setMouseCallback(root.filename, choose_points)
cv2.imshow(root.filename, img)

messagebox.showinfo("how to use:",
                    "HOW TO USE:    click on the four corners of the A4 paper  CAUTION:  normal A4 paper with dimensions 21x30 cm")

cv2.waitKey(0)

pts = np.array(eval("[(cord[0], cord[1]), (cord[2], cord[3]), (cord[4], "       #saves the cordinates
                    "cord[5]), (cord[6], cord[7])]"), dtype="float32")
warped = four_point_transform(img, pts)                                         #makes perpective transformation

img = resizeWithAspectRatio(warped, height=int(height * 0.9))                   # resize the picture to fill the screen
org = img                                                                       # save transformed picture

# filter green
cv2.namedWindow('greenfilter')
cv2.createTrackbar('green_filter', 'greenfilter', 0, 255, nothing)
# cv2.createTrackbar('highbound','greenfilter',0,255,nothing)

cv2.imshow('greenfilter', img)
messagebox.showinfo("how to use:",
                    "HOW TO USE:    adjust the slider     when done press esc")

while (1):  # update picture drawing
    if cv2.waitKey(1) & 0xFF == 27:
        break
    gl = cv2.getTrackbarPos('green_filter', 'greenfilter')
    # gh = cv2.getTrackbarPos('highbound', 'greenfilter')
    greenlow = int(gl + 10)
    # greenhigh = int(gh + 10)
    img2 = greenfilter(img)
    imggreen = img2 & org
    cv2.imshow('greenfilter', imggreen)  # cancel on escape

# cv2.waitKey(0)
# edged = cv2.Canny(imggreen, 30, 200)                                                # Find Canny edges
# contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL,
#                                       cv2.CHAIN_APPROX_NONE)                       # Finding Contours
# edged_inv = cv2.bitwise_not(edged)                                                  # inverted edges
# cv2.drawContours(img, contours, -1, (0, 255, 0), 2)                                 # draw contours
h, w = img.shape[:2]  # get picture dimensions
mask = np.zeros((h + 2, w + 2), np.uint8)

cv2.destroyAllWindows()
cv2.namedWindow('flood')
cv2.setMouseCallback('flood', flood)
cv2.imshow('flood', imggreen)
messagebox.showinfo("how to use:",
                    "HOW TO USE:    click on the black areas that are not eaten by insects. this may "
                    "take up to 10 seconds of processing after click    when done press esc")

while (1):  # update picture drawing
    if cv2.waitKey(1) & 0xFF == 27:
        break
    cv2.imshow('flood', imggreen)
    if floodx > 0:
        cv2.floodFill(imggreen, mask, (floodx, floody), (254, 254, 254))
        imggreen = floodreplace(imggreen)
        # cv2.imshow('flood', imggreen)
        # _, filt = cv2.threshold(imggreen, 250, 255, cv2.THRESH_BINARY)
        # filt = cv2.cvtColor(filt, cv2.COLOR_GRAY2RGB)
        # cv2.imshow('flood', imggreen)
        # cv2.imshow('flood', filt)
        # cv2.waitKey(0)
    # floodfill with black
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)                                 # grayscale
result = imggreen  # black+image
# img = result


cv2.destroyAllWindows()
cv2.namedWindow('test draw')
cv2.createTrackbar('pensize', 'test draw', 0, 255, nothing)
cv2.setMouseCallback('test draw', line_drawing)  # draw on picture
cv2.imshow('test draw', result)
messagebox.showinfo("how to use:",
                    "HOW TO USE: leftclick and hold to draw black, right click and hold to remove."
                    " when done press esc")

while (1):  # update picture drawing
    cv2.imshow('test draw', result)  # cancel on escape
    if cv2.waitKey(1) & 0xFF == 27:
        calcBlackpart(result)

    r = cv2.getTrackbarPos('pensize', 'test draw')
    pencilsize = int(r / 8)

# ***************************      MAIN END   *****************************************


# cv2.destroyAllWindows()
# root.destroy()
# sys.exit()


# the input dialog
# USER_INP = simpledialog.askinteger(title="Test",
#                                   prompt="What'the area of the picture? "
#                                          "[cm^2]:" + "A4=630cm^2")


# cv2.namedWindow("output", cv2.WINDOW_NORMAL)  # Create window with freedom of
# dimensions
# cv2.resizeWindow("output", 400, 300)           # Resize window to specified
# dimensions


# check it out
# print("area is=", USER_INP)
# *********************************************************************************

# *****************   drawing   ***************************************************


# cv2.imshow('points', img)
#    if cv2.waitKey(1) & 0xFF == 27:
#        break


# apply the four point tranform to obtain a "birds eye view" of
# the image


# show the original and warped images
# cv2.imshow("Original", image)
# cv2.imshow("Warped", img)
# cv2.waitKey(0)


# cv2.imshow('greenMask', img)

# pixel = (20,60,80) # some stupid default

# mouse callback function

# def pick_color(event,x,y,flags,param):
#  if event == cv2.EVENT_LBUTTONDOWN:
#     pixel = image_hsv[y,x]

# you might want to adjust the ranges(+-10, etc):
#     upper =  np.array([pixel[0] + 10, pixel[1] + 3, pixel[2] + 40])
#    lower =  np.array([pixel[0] - 10, pixel[1] - 3, pixel[2] - 40])
#    print(pixel, lower, upper)

#   image_mask = cv2.inRange(image_hsv,lower,upper)
#   cv2.imshow("mask",image_mask)

# cv2.namedWindow('test draw')
# cv2.setMouseCallback('test draw', pick_color)
# cv2.imshow('test draw', image_hsv)

# image_hsv =  cv2.cvtColor(img,cv2.COLOR_BGR2HSV)   # global ;
# upper =  np.array([54, 258, 126])
# lower =  np.array([34, 252, 46])
# img = cv2.inRange(image_hsv,lower,upper)

# cv2.imshow("mask",img)
# cv2.waitKey(0)
# [ 34 252  46] [ 54 258 126]

# cv2.imshow('greenMask', img)
# cv2.waitKey(0)


# cv2.waitKey(0)
# cv2.destroyAllWindows()


# cv2.imshow('Canny Edges After Contouring', edged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# cv2.imshow('Canny INV Edges After Contouring', edged_inv)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Draw all contours
# -1 signifies drawing all contours

# cv2.imshow('Contours', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('inv', edged_inv)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# cv2.imshow('result', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# img = np.copy(img)
# img[mask != 0] = [0, 0, 0]

# plt.imshow(masked_image)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.imshow(img, cmap='gray')

# img = (img, 0)

# outputImage = np.where(img == (10,255,10), background, img)
# print(outputImage)

# cv2.imshow('hej', img)


# ************************************************************************


# *****************   Calculate area *************************************
