import math 
import copy
import mpmath

# this looks through the class list output_dict['detection_classes'], and finds the index of the first 3 different box values 
# assuming they are all in order of probability
def find_class_box_index(classArray, labels, scores):
    classArray = classArray.tolist()
    index = [[0, labels[int(classArray[0])]]]
    value = [classArray[0]]
    for number in classArray:
        if number not in value:
            if scores[classArray.index(number)] <0.5:
                print("Warning: Not all reference point values are above 50% accurate")
            index.append([classArray.index(number), labels[int(number)]])
            value.append(number)
            if len(index)==3:
                break
    return index

#finds bounding box centres and puts the top 3 in the centreCoords variable
# output form #[[x1,y1],[x2,y2],[x3,y3]]
def find_bbox_centre(index, bounding_box_array, imW, imH):
    centreCoords=[]
    for i in index:
        dims = bounding_box_array[i[0]]
        centre = [((dims[3]*imW+dims[1]*imW)/2), ((dims[2]*imH+dims[0]*imH)/2)]
        centreCoords.append(centre)
    return centreCoords

#order of the angles is important
# this is taken from the line between the image centre point and the top of the image 
#image axis runs from 0,0 at top left and xmax, ymax bottom right rememeber!
#Function to calculate the angles between centre point pixels of the reference points

def calculate_angles(centreCoords, imageCentre):
    angles = []
    for centre in centreCoords:
        x = abs(centre[0]-imageCentre[0])
        y = abs(centre[1]-imageCentre[1])
        if centre[0]>imageCentre[0] and centre[1]<imageCentre[1]:
            angle = math.atan(x/y)
        elif centre[0]>imageCentre[0] and centre[1]>imageCentre[1]:
            angle = math.atan(y/x)+ math.pi/2
        elif centre[0]<imageCentre[0] and centre[1]>imageCentre[1]:
            angle = math.atan(x/y) + math.pi
        elif centre[0]<imageCentre[0] and centre[1]<imageCentre[1]:
            angle = math.atan(y/x)+ (3/2*math.pi)
        angles.append(angle)
    return angles

# gets data and creates a variable in refCoords which contains the following of the 3 centre
# point infomation [real x value, real y value, angle of reference pint from centre, pixel x centre, pixel y centre] 
def sort_data(index, angles, realCoords, centreCoords):
    refCoords = []
    for i in range(3):
        refCoords.append([realCoords[str(index[i][1])][0], realCoords[str(index[i][1])][1], angles[i], centreCoords[i][0], centreCoords[i][1]])
    refCoords.sort(key=lambda l:l[2])
    # check there are no zero value angles when subtracted
    if refCoords[2][2]-refCoords[0][2]==0:
        refCoords.append(1)
    elif refCoords[1][2]-refCoords[2][2]==0:
        refCoords = [refCoords[2], refCoords[0], refCoords[1]]
        refCoords.append(2)
    elif refCoords[0][2]-refCoords[1][2]==0:
        refCoords = [refCoords[1], refCoords[2], refCoords[0]]
        refCoords.append(3)
    return refCoords


#ToTal Algorithm 
# refCoords = [[x1,y1], [x2,y2],[x3,y3]]
# angles = [a1,a2,a3]
def ToTal_Algorithm(refCoords):
    # compute modified beacon coordinates
    x_1 = refCoords[0][0] - refCoords[1][0]
    y_1 = refCoords[0][1] - refCoords[1][1]
    x_3 = refCoords[2][0] - refCoords[1][0]
    y_3 = refCoords[2][1] - refCoords[1][1]
    
    # compute three cot(.)
    T12 = mpmath.cot(refCoords[1][2]-refCoords[0][2]) 
    T23 = mpmath.cot(refCoords[2][2]-refCoords[1][2])
    T31 = (1 - T12*T23)/(T12 + T23)
    
    #compute modified circle center coordinates 
    x_12 = x_1 + T12*y_1
    y_12 = y_1 - T12*x_1
    x_23 = x_3 - T23*y_3
    y_23 = y_3 + T23*x_3
    x_31 = (x_3 + x_1) + T31*(y_3 - y_1)
    y_31 = (y_3 + y_1) - T31*(x_3 - x_1)
    
    # compute k_31
    k_31 = x_1*x_3 + y_1*y_3 + T31*(x_1*y_3 - x_3*y_1)
    
    #compute D
    D = (x_12 - x_23)*(y_23 - y_31) - (y_12 - y_23)*(x_23 - x_31)
    
    #compute the robot position
    xR = refCoords[1][0] + (k_31*(y_12 - y_23))/D
    yR = refCoords[1][1] + (k_31*(x_23 - x_12))/D
    return xR, yR


# ToTal algorithm in the case when the subtraction of two of the angles = 0 or pi 
def ToTal_Algorithm_Special_Cases(refCoords):
    # compute modified beacon coordinates
    x_i = refCoords[0][0] - refCoords[1][0]
    y_i = refCoords[0][1] - refCoords[1][1]
    x_k = refCoords[2][0] - refCoords[1][0]
    y_k = refCoords[2][1] - refCoords[1][1]
    
    # compute Tij
    Tij = mpmath.cot(refCoords[1][2] - refCoords[0][2])  
    
    # compute the modified circle center coordinates
    x_ij = x_i + Tij*y_i
    y_ij = y_i - Tij*x_i
    x_jk = x_k + Tij*y_k
    y_jk = y_k - Tij*x_k
    x_ki = y_k - y_i
    y_ki = x_i - x_k
    
    # compute k_ki
    k_ki = x_i*y_k - x_k*y_i
    
    #compute D
    D=(x_jk - x_ij)*y_ki + (y_ij - y_jk)*x_ki
    
    # compute the robot position
    xR = refCoords[1][0] + (k_ki*(y_ij - y_jk))/D
    yR = refCoords[1][1] + (k_ki*(x_jk - x_ij))/D
    return xR, yR

# this function will take a list of reference points locate the largest and minimum
# x coords and from this the angle between these two points will be calculated
def z_depth_calculator(refCoords):
     #order the reference in size according to the x values
     refCoords.sort(key=lambda l:l[3])
     # calculate the angle between the two chosen reference points using pixel values
     x = refCoords[2][3]-refCoords[0][3]
     y = abs(refCoords[2][4]-refCoords[0][4])
     pixelAngle = math.atan(x/y)
     # calculate the actual distance between two reference points
     rx = abs(refCoords[2][0]-refCoords[0][0])
     ry = abs(refCoords[2][1]-refCoords[0][1])
     actualDistance = math.sqrt(rx*rx + ry*ry)
     # resolve the actual distance in the x direction of the image
     imageXDistance = actualDistance * math.sin(pixelAngle)
     #work out the distance per pixel and multiply up by the image width size to get actual image distance
     disPerPixel = imageXDistance/x
     actualImageXdistance = disPerPixel*640
     #use the scale factor of the length of the image at the focal distance to caluclate the depth 
     zDepth = 88.5/92 * actualImageXdistance
     return zDepth
     
     
     
     
